"""Live web dashboard.  .venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.web RUN [--port 8740]"""
import argparse
from dataclasses import asdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import subprocess
from pathlib import Path
from statistics import median
import time
import threading
from urllib.parse import parse_qs, urlsplit

from .health import Health

import monitoring
from monitoring.telemetry import actors, lanes, queues

from .dashboard import snapshot
from .reporting import baselines, by_deck, deck_of, jsonl, outcomes, paired, pool, rate, unpaired

PAGE = Path(__file__).with_name('web.html')
TELEMETRY_PANEL = Path(monitoring.__file__).with_name('telemetry.js')
WINDOW = 5      # training trend: last WINDOW batches pooled vs the WINDOW before
EVAL_BACK = 3   # evaluation trend: latest vs the evaluation EVAL_BACK earlier (same seeds, paired)
LABELS = dict(mcts20k='MCTS20k', puct='PUCT')


finished = lru_cache(maxsize=None)(outcomes)  # completed result files never change


def losses(logs):
    """Final train-step losses from the run logs; each loss line precedes its iteration's log row."""
    out, pending = {}, None
    for path in sorted(logs.glob('*.log'), key=lambda f: f.stat().st_mtime):
        for line in path.read_text(errors='replace').splitlines():
            m = re.search(r'step (\d+)/(\d+): value ([\d.]+) policy ([\d.]+)', line)
            if m and m[1] == m[2]:
                pending = dict(value=float(m[3]), policy=float(m[4]))
            elif line.startswith('{"iteration"') and pending:
                out[json.loads(line)['iteration']] = pending
                pending = None
    return out


@lru_cache(maxsize=None)
def gpu_summary(path):
    util, mem = [], []
    for line in Path(path).read_text().splitlines()[1:]:
        parts = line.split(', ')
        if len(parts) == 5:
            util.append(float(parts[1]))
            mem.append(float(parts[3]))
    return dict(util=sum(util) / len(util), vram=max(mem) / 1024) if util else {}


def gpu_stats(worker_log, tail=8192):
    """Last two cumulative gpu_stats records from a stage's worker.log (NN evaluations = leaves)."""
    with open(worker_log, 'rb') as f:
        f.seek(max(0, f.seek(0, 2) - tail))
        lines = f.read().decode(errors='replace').splitlines()
    stats = []
    for line in lines:
        if line.startswith('gpu_stats: '):
            try:
                stats.append(json.loads(line[11:]))
            except json.JSONDecodeError:
                pass
    return stats[-2:]


def cpu_fallback(stats):
    # GPU retries are already counted in leaves; oversized rows are not.
    if not all(k in stats for k in ('leaves', 'cpu_leaves', 'cpu_precision_retries')):
        return None
    requests = stats['leaves'] + stats['cpu_leaves']
    return (stats['cpu_leaves'] + stats['cpu_precision_retries']) / requests if requests else None


@lru_cache(maxsize=None)
def inference_summary(worker_log):
    stats = gpu_stats(worker_log)
    last = stats[-1] if stats else {}
    return dict(nn=last['leaves'] / last['seconds'] if last.get('seconds') else None,
                cpu_fallback=cpu_fallback(last))


def system():
    out = {}
    try:
        u, used, total = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                        timeout=2).stdout.split('\n')[0].split(', ')
        out.update(gpu=float(u), vram=float(used) / 1024, vram_total=float(total) / 1024)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    mem = dict(line.split(':') for line in Path('/proc/meminfo').read_text().splitlines())
    kib = lambda k: float(mem[k].split()[0]) / 1024 ** 2
    out.update(ram=kib('MemTotal') - kib('MemAvailable'), ram_total=kib('MemTotal'))
    return out


def comparison(out, tests, test_results):
    """Historical baselines, first and latest evaluation on the starts they all share (paired)."""
    base = baselines(out)
    cols = {LABELS.get(k, k): v for k, v in base.items()}
    for i in dict.fromkeys(tests[:1] + tests[-1:]):
        cols[f'iter {i}'] = test_results[i]
    keys = set.intersection(*map(set, cols.values())) if len(cols) > 1 else set()
    if not keys:
        return None
    groups = {'all': sorted(keys)}
    for k in sorted(keys):
        groups.setdefault(deck_of(k), []).append(k)
    return dict(columns=list(cols), n=len(keys),
                rows=[dict(deck=d, rates={c: rate(sum(v[k] for k in ks), len(ks)) for c, v in cols.items()})
                      for d, ks in groups.items()],
                diffs={a: {b: {d: paired(cols[a], cols[b], ks) for d, ks in groups.items()} for b in cols if b != a}
                       for a in cols})


def api(out):
    data = snapshot(out)
    training = out / 'training'
    history = jsonl(training / 'log.jsonl')
    progress = data['progress']
    its = sorted(r['iteration'] for r in history if r.get('iteration', 0) > 0 and 'games' in r)
    tests = sorted({r['test'] for r in history if 'test' in r})
    train = [dict(it=i, rates=by_deck(finished(training / f'iter{i:03d}/results.jsonl'))) for i in its]
    test_results = {i: finished(training / f'test/iter{i:03d}/results.jsonl') for i in tests}
    test = [dict(it=i, rates=by_deck(r)) for i, r in test_results.items()]
    loss = losses(out.parent / 'logs')
    timing = {r['iteration']: r for r in history if r.get('play_seconds')}
    for b in train:
        r, gpu, worker = timing.get(b['it'], {}), training / f'iter{b["it"]:03d}/gpu.csv', training / f'iter{b["it"]:03d}/worker.log'
        b.update(loss=loss.get(b['it']), play=r.get('play_seconds'), fit=r.get('train_seconds'),
                 gps=r['games'] / r['play_seconds'] if r else None, gpu=gpu_summary(gpu) if gpu.exists() else {},
                 **(inference_summary(worker) if worker.exists() else dict(nn=None, cpu_fallback=None)))
    decks = sorted({d for b in train + test for d in b['rates'] if d != 'all'})

    trend = []
    recent, before = train[-WINDOW:], train[-2 * WINDOW:-WINDOW]
    new_eval, old_eval = (tests[-1], tests[-1 - EVAL_BACK]) if len(tests) > EVAL_BACK else (None, None)
    eval_groups = {}
    if new_eval is not None:
        a, b = test_results[new_eval], test_results[old_eval]
        for k in a.keys() & b.keys():
            eval_groups.setdefault(deck_of(k), []).append(k)
        eval_groups['all'] = [k for keys in eval_groups.values() for k in keys]
    for deck in ['all'] + decks:
        row = dict(deck=deck, train=pool(recent, deck) if recent else None)
        if len(before) == WINDOW:
            row['train_d'] = unpaired(row['train'], pool(before, deck))
        if new_eval is not None:
            keys = eval_groups.get(deck, [])
            row['eval'] = rate(sum(a[k] for k in keys), len(keys))
            row['eval_d'] = paired(a, b, keys)
        trend.append(row)

    live, speed = None, {}
    typical = [r['games'] / r['play_seconds'] for r in history if r.get('play_seconds')]
    speed['typical'] = median(typical) if typical else None
    nn = [b['nn'] for b in train if b['nn']]
    speed['nn_typical'] = median(nn) if nn else None
    if train:
        speed.update(cpu_fallback=train[-1]['cpu_fallback'], fallback_source=f'iteration {train[-1]["it"]} self-play')
    if data['state'] in ('running', 'stopping') and progress.get('phase') == 'play' and progress.get('directory'):
        directory = Path(progress['directory'])
        results = outcomes(directory / 'results.jsonl')
        live = dict(test='/test/' in str(directory), done=len(results), total=progress['total'],
                    rates=by_deck(results))
        starts = directory / 'starts.jsonl'
        elapsed = time.time() - (starts.stat().st_mtime if starts.exists() else progress['time'])
        speed['stage'] = len(results) / elapsed if elapsed > 0 else None
        stats = gpu_stats(directory / 'worker.log') if (directory / 'worker.log').exists() else []
        if stats:
            speed.update(cpu_fallback=cpu_fallback(stats[-1]),
                         fallback_source='this evaluation' if live['test'] else 'this self-play')
        if len(stats) == 2 and stats[1]['seconds'] > stats[0]['seconds']:
            speed['nn'] = (stats[1]['leaves'] - stats[0]['leaves']) / (stats[1]['seconds'] - stats[0]['seconds'])
        if speed['typical'] and not live['test']:
            speed['eta'] = max(0, progress['total'] / speed['typical'] - elapsed)
    return dict(state=data['state'], run=data['config']['run']['id'], progress=progress, live=live,
                speed=speed, system=system(), train=train, test=test, decks=decks, trend=trend,
                compare=comparison(out, tests, test_results),
                windows=dict(train=WINDOW, eval=[new_eval, old_eval]))


def playing(states):
    """A route serving one run_actors component of the playing stage; none outside play."""
    def route(out):
        progress = json.loads((out / 'training/status.json').read_text())
        if progress['phase'] != 'play':
            return []
        return [asdict(s) for s in states(Path(progress['directory']) / 'telemetry.jsonl')]
    return route


ROUTES = {'/api': api, '/api/actors': playing(actors), '/api/queues': playing(queues), '/api/lanes': playing(lanes)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, nargs='?')
    p.add_argument('--port', type=int, default=8740)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--mock', choices=('normal', 'fallback', 'stale', 'missing'))
    a = p.parse_args()
    if bool(a.run) == bool(a.mock):
        p.error('provide either RUN or --mock SCENARIO')
    out, sampler = None, None
    if a.mock:
        from .health_mock import MockHealth
        health = MockHealth(a.mock)
    else:
        out = (a.run / 'out' if (a.run / 'out').is_dir() else a.run).resolve()
        health = Health(out)
        sampler = threading.Thread(target=health.run, name='dashboard-health', daemon=True)
        sampler.start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                route = urlsplit(self.path)
                if route.path == '/api/health':
                    window = int(parse_qs(route.query).get('window', ['900'])[0])
                    body = json.dumps(health.view(window), allow_nan=False).encode()
                elif route.path in ROUTES:
                    if out is None:
                        result = dict(error='Quality view unavailable in mock mode') if route.path == '/api' else []
                    else:
                        result = ROUTES[route.path](out)
                    body = json.dumps(result).encode()
                elif route.path == '/telemetry.js':
                    body = TELEMETRY_PANEL.read_bytes()
                elif route.path == '/health.js':
                    body = PAGE.with_name('health.js').read_bytes()
                elif route.path == '/health.css':
                    body = PAGE.with_name('health.css').read_bytes()
                elif route.path == '/':
                    body = PAGE.read_bytes()
                else:
                    self.send_error(404)
                    return
            except Exception as e:  # files mid-write or run not initialized: report, keep serving
                body = json.dumps(dict(error=repr(e))).encode()
            self.send_response(200)
            content_type = ('application/json' if self.path.startswith('/api') else
                            'text/javascript' if route.path in ('/health.js', '/telemetry.js') else
                            'text/css' if self.path.startswith('/health.css') else 'text/html')
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f'http://{a.host}:{a.port}/  ({out or "MOCK: " + a.mock})', flush=True)
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    try:
        server.serve_forever()
    finally:
        if sampler:
            health.stop.set()
            sampler.join(timeout=3)
        server.server_close()


if __name__ == '__main__':
    main()
