"""Live web dashboard.  .venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.web RUN [--port 8740]"""
import argparse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import re
import subprocess
from pathlib import Path
from statistics import median
import time

from apps.gumbel_nn_combat_expert_iteration.dashboard import jsonl, snapshot

PAGE = Path(__file__).with_name('web.html')
WINDOW = 5      # training trend: last WINDOW batches pooled vs the WINDOW before
EVAL_BACK = 3   # evaluation trend: latest vs the evaluation EVAL_BACK earlier (same seeds, paired)
Z = 1.96


def deck_of(fight_id):
    parts = fight_id.split(':')  # training: ns:iter:deck:i; evaluation: ns:deck:split:i
    return parts[2] if parts[1].isdigit() else parts[1]


def wilson(w, n):
    if not n:
        return [0, 0]
    p, d = w / n, 1 + Z * Z / n
    c, h = (p + Z * Z / (2 * n)) / d, Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return [c - h, c + h]


def rate(w, n):
    return dict(w=w, n=n, p=w / n if n else None, ci=wilson(w, n))


def outcomes(path):
    return {r['fight_id']: bool(r['won']) for r in jsonl(path)}


finished = lru_cache(maxsize=None)(outcomes)  # completed result files never change


def by_deck(results):
    decks = {}
    for k, won in results.items():
        decks.setdefault(deck_of(k), []).append(won)
    out = {d: rate(sum(v), len(v)) for d, v in sorted(decks.items())}
    out['all'] = rate(sum(results.values()), len(results))
    return out


def unpaired(new, old):
    p1, n1, p2, n2 = new['p'], new['n'], old['p'], old['n']
    d, se = p1 - p2, math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return dict(d=d, lo=d - Z * se, hi=d + Z * se)


def paired(new, old, keys):
    if not keys:
        return None  # training-only decks need not belong to the unseen validation panel
    diffs = [new[k] - old[k] for k in keys]
    n = len(diffs)
    d = sum(diffs) / n
    se = math.sqrt(sum((x - d) ** 2 for x in diffs) / (n - 1) / n) if n > 1 else 0
    return dict(d=d, lo=d - Z * se, hi=d + Z * se)


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


def pool(batches, deck):
    w = sum(b['rates'][deck]['w'] for b in batches if deck in b['rates'])
    n = sum(b['rates'][deck]['n'] for b in batches if deck in b['rates'])
    return rate(w, n)


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
        r, gpu = timing.get(b['it'], {}), training / f'iter{b["it"]:03d}/gpu.csv'
        b.update(loss=loss.get(b['it']), play=r.get('play_seconds'), fit=r.get('train_seconds'),
                 gps=r['games'] / r['play_seconds'] if r else None, gpu=gpu_summary(gpu) if gpu.exists() else {})
    decks = sorted({d for b in train + test for d in b['rates'] if d != 'all'})

    trend = []
    recent, before = train[-WINDOW:], train[-2 * WINDOW:-WINDOW]
    new_eval, old_eval = (tests[-1], tests[-1 - EVAL_BACK]) if len(tests) > EVAL_BACK else (None, None)
    for deck in ['all'] + decks:
        row = dict(deck=deck, train=pool(recent, deck) if recent else None)
        if len(before) == WINDOW:
            row['train_d'] = unpaired(row['train'], pool(before, deck))
        if new_eval is not None:
            a, b = test_results[new_eval], test_results[old_eval]
            keys = [k for k in a if k in b and (deck == 'all' or deck_of(k) == deck)]
            row['eval'] = rate(sum(a[k] for k in keys), len(keys))
            row['eval_d'] = paired(a, b, keys)
        trend.append(row)

    live, speed = None, {}
    typical = [r['games'] / r['play_seconds'] for r in history if r.get('play_seconds')]
    speed['typical'] = median(typical) if typical else None
    if progress.get('phase') == 'play' and progress.get('directory'):
        directory = Path(progress['directory'])
        results = outcomes(directory / 'results.jsonl')
        live = dict(test='/test/' in str(directory), done=len(results), total=progress['total'],
                    rates=by_deck(results))
        starts = directory / 'starts.jsonl'
        elapsed = time.time() - (starts.stat().st_mtime if starts.exists() else progress['time'])
        speed['stage'] = len(results) / elapsed if elapsed > 0 else None
        if speed['typical'] and not live['test']:
            speed['eta'] = max(0, progress['total'] / speed['typical'] - elapsed)
    return dict(state=data['state'], run=data['config']['run']['id'], progress=progress, live=live,
                speed=speed, system=system(), train=train, test=test, decks=decks, trend=trend,
                windows=dict(train=WINDOW, eval=[new_eval, old_eval]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--port', type=int, default=8740)
    p.add_argument('--host', default='127.0.0.1')
    a = p.parse_args()
    out = (a.run / 'out' if (a.run / 'out').is_dir() else a.run).resolve()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            try:
                body = json.dumps(api(out)).encode() if self.path.startswith('/api') else PAGE.read_bytes()
            except Exception as e:  # files mid-write or run not initialized: report, keep serving
                body = json.dumps(dict(error=repr(e))).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json' if self.path.startswith('/api') else 'text/html')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    print(f'http://{a.host}:{a.port}/  ({out})', flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
