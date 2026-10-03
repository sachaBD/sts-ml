#!/usr/bin/env python3
"""Replay recorded Act 2 boss fights at several MCTS simulation budgets (experiments/act2-boss-search).

Fights come from an overworld_v1 run: each Act 2 boss fight's pre-fight macro state is the `after` / `state` of
the step just before it. The fight is rebuilt with the original run seed and floor (apps/boss_rebuild/worker.cpp)
and played with guided-rollout MCTS (8 particles) once per budget. The recorded original result (20k, real
state) is kept alongside for a rebuild-fidelity check.

  bench.py --source RUN_OUT --out OUT --worker BIN --per-boss 100 --sims 20000 100000 --workers 10 [--limit N]

out/results.jsonl: one line per (fight, budget); out/fights.jsonl: the sampled fights; out/summary.json.
"""
import argparse
import json
import random
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import duckdb

BOSSES = ("automaton", "champ", "collector")


def act2_boss_fights(source):
    db = duckdb.connect()
    rows = db.execute(f"select run_key, step_index, kind, record_json from read_parquet('{source}/steps-*.parquet') "
                      "order by run_key, step_index").fetchall()
    runs = {}
    for run_key, _, kind, record in rows:
        runs.setdefault(run_key, []).append((kind, json.loads(record)))
    fights = []
    for run_key, steps in runs.items():
        for i, (kind, x) in enumerate(steps):
            if kind != 'fight' or x['category'] != 'boss' or x['state']['act'] != 2:
                continue
            prev = steps[i - 1][1]
            pre = prev.get('after') or prev.get('state')
            if not pre or 'deck' not in pre:
                continue
            pre = {k: v for k, v in pre.items() if k != 'map'}
            fights.append({'fight_id': x['fight_id'], 'seed': int(run_key.rsplit(':', 1)[1]), 'encounter': x['encounter'],
                           'floor': pre['floor'] + 1, 'hp_before': x['hp_before'], 'state': pre,
                           'orig_won': x['won'], 'orig_final_hp': x['state']['hp'],
                           'orig_potions_start': len(pre['potions']), 'orig_potions_end': len(x['state']['potions'])})
    return fights


def play(worker, fight, sims):
    req = {'seed': fight['seed'], 'ascension': 20, 'act': 2, 'floor': fight['floor'], 'encounter': fight['encounter'],
           'hp_before': fight['hp_before'], 'simulations': sims, 'state': fight['state']}
    with tempfile.NamedTemporaryFile('w', suffix='.json') as f:
        json.dump(req, f)
        f.flush()
        done = subprocess.run([worker, f.name], capture_output=True, text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--worker', required=True)
    ap.add_argument('--per-boss', type=int, default=100)
    ap.add_argument('--sims', type=int, nargs='+', default=[20000, 100000])
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, help='first N sampled fights only (smoke)')
    ap.add_argument('--sample-seed', type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    fights = act2_boss_fights(a.source)
    rng = random.Random(a.sample_seed)
    sample = []
    for boss in BOSSES:
        pool = sorted((f for f in fights if f['encounter'] == boss), key=lambda f: f['fight_id'])
        sample += rng.sample(pool, min(a.per_boss, len(pool)))
    rng.shuffle(sample)  # mixes bosses so a partial run is balanced
    if a.limit:
        sample = sample[:a.limit]
    with open(a.out / 'fights.jsonl', 'w') as f:
        for x in sample:
            f.write(json.dumps(x) + '\n')

    # Most expensive budget first per fight keeps the tail short.
    jobs = [(x, s) for x in sample for s in sorted(a.sims, reverse=True)]
    jobs.sort(key=lambda j: -j[1])
    t0 = time.monotonic()
    n = 0
    with ThreadPoolExecutor(a.workers) as pool, open(a.out / 'results.jsonl', 'w') as out:
        futures = {pool.submit(play, a.worker, x, s): (x, s) for x, s in jobs}
        for fut in as_completed(futures):
            x, s = futures[fut]
            r = fut.result()
            out.write(json.dumps({'fight_id': x['fight_id'], 'encounter': x['encounter'], 'sims': s, **r}) + '\n')
            out.flush()
            n += 1
            print(f'{n}/{len(jobs)} {time.monotonic() - t0:.0f}s {x["encounter"]} sims={s} won={r["won"]} '
                  f'{r["seconds"]:.0f}s', flush=True)
    (a.out / 'summary.json').write_text(json.dumps({'fights': len(sample), 'sims': a.sims, 'jobs': len(jobs),
                                                    'wall_seconds': time.monotonic() - t0}, indent=2))


if __name__ == '__main__':
    main()
