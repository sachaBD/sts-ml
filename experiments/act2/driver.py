#!/usr/bin/env python3
"""Act 1+2 overworld expert iteration. Every stage is a managed run (runs.run) with frozen inputs; resumable.

Round i: collect --batch runs with the incumbent (eps, random routes) -> train a candidate on the last --window
collections (target "floors") -> paired gate vs the incumbent on fixed dev seeds (greedy). Final: fresh seeds,
selected vs initial. Combat: guided-rollout MCTS throughout. Run from repo root with PYTHONPATH=.
"""
import argparse
import datetime as dt
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from runs.run import RUNS
from agents.overworld.value.core import read_runs, run_score, acts_cleared

PYTHON = sys.executable
EXP = Path('experiments/act2')
SIMS = '500,5000,10000,10000,20000'
DECIDE = ['rest', 'path', 'shop', 'event', 'neow', 'boss_relic']
DEADLINE = float('inf')
DEV_SEED, COLLECT_SEED, FRESH_SEED = 960_000_000_000, 970_000_000_000, 980_000_000_000


def log(message):
    line = f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {message}"
    print(line, flush=True)
    with (EXP / 'RUNBOOK.md').open('a') as f:
        f.write(line + '\n')


def save(path, obj):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, indent=2)); tmp.replace(path)


def job(schema, name, cmd, inputs=()):
    matches = list(RUNS.glob(f'schema={schema}/date=*/id={name}/run.json'))
    if matches:
        meta = json.loads(matches[0].read_text())
        if meta['status'] != 'done':
            raise RuntimeError(f'incomplete job {name}: inspect {matches[0]}')
        return matches[0].parent / 'out'
    argv = [PYTHON, '-m', 'runs.run', schema, name, '--no-compact']
    for source in inputs:
        argv += ['--input', str(source)]
    argv += ['--', *map(str, cmd)]
    log('START ' + name)
    remaining = DEADLINE - time.time()
    if remaining <= 0:
        raise TimeoutError('deadline reached')
    proc = subprocess.Popen(argv, start_new_session=True)
    try:
        result = proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait()
        raise TimeoutError('deadline reached; partial child outputs preserved')
    if result:
        raise subprocess.CalledProcessError(result, argv)
    log('DONE ' + name)
    return next(RUNS.glob(f'schema={schema}/date=*/id={name}/run.json')).parent / 'out'


def play(name, worker, ckpt, first, seeds, workers, eps=0.0, route_p=0.0):
    cmd = [PYTHON, 'apps/run_rl/play.py', '--out', '{out}', '--first-seed', first, '--seeds', seeds,
           '--workers', workers, '--worker', worker, '--policy', 'net', '--ckpt', ckpt, '--max-act', 2,
           '--target', 'floors', '--decide', *DECIDE, '--sims', SIMS, '--overworld-record', '--collection-id', name,
           '--eps', eps, '--route-p', route_p]
    return job('overworld_v1', name, cmd, [ckpt])


def stats(runs):
    n = len(runs)
    sc = [run_score(r, 'floors') for r in runs]
    ac = [acts_cleared(r) for r in runs]
    return {'n': n, 'score': statistics.mean(sc), 'act1': sum(a >= 1 for a in ac) / n,
            'act2': sum(a >= 2 for a in ac) / n, 'floor': statistics.mean(r['floor'] for r in runs)}


def paired(a, b):
    x = {r['seed']: r for r in read_runs([a])}; y = {r['seed']: r for r in read_runs([b])}
    if set(x) != set(y):
        raise RuntimeError('paired evaluation seed mismatch')
    seeds = sorted(x); n = len(seeds)
    se = lambda z: statistics.stdev(z) / math.sqrt(n)
    out = {'candidate': stats(list(x.values())), 'incumbent': stats(list(y.values()))}
    for key, f in [('score', lambda r: run_score(r, 'floors')), ('act1', lambda r: float(acts_cleared(r) >= 1)),
                   ('act2', lambda r: float(acts_cleared(r) >= 2)), ('floor', lambda r: float(r['floor']))]:
        d = [f(x[s]) - f(y[s]) for s in seeds]
        out[key + '_diff'], out[key + '_se'] = statistics.mean(d), se(d)
    return out


def report(state, out):
    lines = ['# Act 2 expert iteration — controller report', '', f"Status: {state['status']}",
             f"Elapsed: {(time.time() - state['started']) / 3600:.2f} h", f"Incumbent: `{state['overworld']}`", '',
             'Score = 0.4 floor/33 + 0.2 [act 1 boss beaten] + 0.4 [act 2 boss beaten]. ± = 1 SE of paired diffs.', '',
             '| Comparison | n | Cand score | Inc score | Δscore ± SE | Δact1 ± SE | Δact2 ± SE | Promoted |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in state['comparisons']:
        lines.append(f"| {r['name']} | {r['candidate']['n']} | {r['candidate']['score']:.3f} | {r['incumbent']['score']:.3f}"
                     f" | {r['score_diff']:+.3f} ± {r['score_se']:.3f} | {r['act1_diff']*100:+.1f} ± {r['act1_se']*100:.1f} pp"
                     f" | {r['act2_diff']*100:+.1f} ± {r['act2_se']*100:.1f} pp | {r.get('promoted', '—')} |")
    lines += ['', '| Play | n | score | act1 clear | act2 clear | mean floor |', '|---|---:|---:|---:|---:|---:|']
    for name, s in state['plays'].items():
        lines.append(f"| {name} | {s['n']} | {s['score']:.3f} | {s['act1']:.1%} | {s['act2']:.1%} | {s['floor']:.1f} |")
    if 'error' in state:
        lines += ['', 'Failure: `' + state['error'] + '`']
    (EXP / 'CONTROLLER.md').write_text('\n'.join(lines) + '\n')
    save(out / 'summary.json', state)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--worker', required=True, type=Path)
    p.add_argument('--init', required=True, type=Path, help='initial overworld checkpoint')
    p.add_argument('--hours', type=float, default=8)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--batch', type=int, default=1000)
    p.add_argument('--eval-seeds', type=int, default=300)
    p.add_argument('--fresh-seeds', type=int, default=400)
    p.add_argument('--final-reserve', type=float, default=1.5, help='hours kept for the fresh final comparison')
    p.add_argument('--round-hours', type=float, default=2.5, help='expected hours per round (stop starting rounds)')
    p.add_argument('--window', type=int, default=3)
    p.add_argument('--eps', type=float, default=0.1)
    p.add_argument('--route-p', type=float, default=0.25)
    p.add_argument('--tag', default='act2')
    p.add_argument('--prior-data', nargs='*', default=[], help='earlier collection out dirs (continuation controller)')
    p.add_argument('--fresh-initial', type=Path, help='existing fresh-seed play of --init (skip re-evaluating it)')
    p.add_argument('--first-round', type=int, default=0, help='round numbering (and collection seeds) start here')
    p.add_argument('--init-trained', action='store_true', help='--init already uses the floors target (TD from round 0)')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    statefile = a.out / 'state.json'
    state = json.loads(statefile.read_text()) if statefile.exists() else {
        'started': time.time(), 'status': 'running', 'initial': str(a.init), 'overworld': str(a.init),
        'collections': list(a.prior_data), 'comparisons': [], 'plays': {}, 'round': a.first_round, 'act2_trained': a.init_trained}
    worker = a.out / 'run_rl_worker'
    if not worker.exists():
        shutil.copy2(a.worker, worker)
    global DEADLINE
    DEADLINE = state['started'] + a.hours * 3600
    persist = lambda: (save(statefile, state), report(state, a.out))
    persist()
    t = a.tag
    try:
        def evaluated(name, ckpt, first, seeds):
            out = play(name, worker, ckpt, first, seeds, a.workers)
            state['plays'][name] = stats(read_runs([out])); persist()
            return out
        while time.time() < DEADLINE - (a.final_reserve + a.round_hours) * 3600:
            i = state['round']; inc = Path(state['overworld'])
            name = f'{t}-r{i:02d}-collect'
            out = play(name, worker, inc, COLLECT_SEED + i * 100_000, a.batch, a.workers, a.eps, a.route_p)
            if str(out) not in state['collections']:
                state['collections'].append(str(out)); state['plays'][name] = stats(read_runs([out]))
            persist()
            data = state['collections'][-a.window:]
            # The first floors-target model bootstraps nothing from an act1-target init: Monte Carlo targets.
            lam = '.7' if state['act2_trained'] else '1'
            trained = job('value_net_v1', f'{t}-r{i:02d}-train', [PYTHON, 'apps/run_rl/train.py', '--out', '{out}/model.pt',
                          '--init', inc, '--data', *data, '--target', 'floors', '--lam', lam, '--decay', '.85',
                          '--epochs', 20], [inc, *data])
            cand = trained / 'model.pt'
            # Dev plays are named by model (shared across controllers): a model's dev play is reused, not replayed.
            base = evaluated(f'act2-dev-{inc.parent.parent.name.removeprefix("id=")}', inc, DEV_SEED, a.eval_seeds)
            test = evaluated(f'act2-dev-{t}-r{i:02d}-train', cand, DEV_SEED, a.eval_seeds)
            r = paired(test, base); r['name'] = f'round {i}'
            # Broad-progress gate: promote on a positive score difference not contradicted by act-1 clears.
            r['promoted'] = r['score_diff'] > 0.5 * r['score_se'] and r['act1_diff'] > -2 * r['act1_se']
            if not state['act2_trained'] and r['score_diff'] > -r['score_se']:
                r['promoted'] = True  # first floors-target model: adopt unless clearly worse (target alignment)
            state['comparisons'].append(r)
            if r['promoted']:
                state['overworld'] = str(cand); state['act2_trained'] = True
            state['round'] += 1; log('GATE ' + json.dumps(r)); persist()
        if state['overworld'] != state['initial']:
            fs = evaluated(f'{t}-fresh-final', Path(state['overworld']), FRESH_SEED, a.fresh_seeds)
            fi = a.fresh_initial or evaluated(f'{t}-fresh-initial', Path(state['initial']), FRESH_SEED, a.fresh_seeds)
            r = paired(fs, fi); r['name'] = 'FRESH selected vs initial'; state['comparisons'].append(r)
            log('FRESH ' + json.dumps(r))
        state['status'] = 'done'
    except Exception as e:
        state['status'] = 'failed'; state['error'] = repr(e); log('FAILED ' + repr(e)); persist(); raise
    persist()


if __name__ == '__main__':
    main()
