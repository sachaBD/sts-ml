"""S0 screen for the human-deck corpus: 4-seed MCTS20k probe, then 16 more seeds on affordable survivors.

Selection: archetype-stratified farthest-first diversity over the frozen training pool (apps.corpus.splits),
skipping decks already screened in human-deck-library-v1. Screen fights later double as teacher bootstrap data.
Exclusions are scope, not difficulty: native errors/timeouts (software) and mean probe seconds > --max-seconds (cost).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import statistics
from pathlib import Path

import pyarrow.parquet as pq

from apps.human_champ.library import CARDS, bucket, distance, family, json_file, results
from apps.run_rl.combat_loop import play, sha, write_starts
from apps.run_rl.single_deck import generate, wilson

LIBRARY = Path('runs/schema=combat_v4/date=2026-10-05/id=human-deck-library-v1/out')
BENCH = Path('runs/schema=human_champ_bench_v1/date=2026-10-05/id=v1/out/starts.parquet')
UNSUPPORTED = {'scrawl'}
PROBE, TOTAL = 4, 20


def select(pool, already, per_bucket):
    chosen = list(already)
    picked = []
    used = {family(r) for r in chosen}
    for group in ['block', 'demon_form', 'exhaust', 'strength', 'mixed']:
        for _ in range(per_bucket):
            avail = [r for r in pool if bucket(r) == group and family(r) not in used
                     and not ({CARDS[c['id']] for c in r['start']['deck']} & UNSUPPORTED)]
            if not avail:
                break
            rank = lambda r: (min((distance(r, s) for s in chosen), default=1.0),
                              hashlib.sha256(r['deck_id'].encode()).hexdigest())
            row = max(avail, key=rank)
            chosen.append(row); picked.append(row); used.add(family(row))
    return picked


def catalogue(out, decks, max_seconds):
    found = {**results(out / 'probe/results.jsonl'), **results(out / 'fill/results.jsonl')}
    rows = []
    for d in decks:
        recs = [found[f] for f in d['fight_ids'] if f in found]
        probe = [found[f] for f in d['fight_ids'][:PROBE] if f in found]
        done = [r for r in recs if r['status'] == 'completed']
        wins = sum(r['fight']['won'] for r in done)
        secs = statistics.mean(r['seconds'] for r in probe) if probe else None
        if len(probe) < PROBE:
            state = 'pending'
        elif any(r['status'] != 'completed' for r in probe):
            state = 'software-excluded'
        elif secs > max_seconds:
            state = 'cost-excluded'
        elif len(recs) < TOTAL:
            state = 'filling'
        else:
            state = 'screened'
        rows.append(dict(d, logged=len(recs), completed=len(done), wins=wins, probe_seconds=secs, state=state,
                         statuses={s: sum(r['status'] == s for r in recs) for s in {r['status'] for r in recs}},
                         confidence_95=wilson(wins, len(done))))
    return rows


def status(run):
    out = run / 'out'
    meta = json_file(run / 'run.json') or {}
    decks = json_file(out / 'decks.json')
    config = json_file(out / 'config.json')
    if not decks:
        print('screen initializing'); return
    rows = catalogue(out, decks, config['max_seconds'])
    print(f'S0 SCREEN {run.name}: {meta.get("status", "?").upper()}  decks {len(rows)}  '
          f'states {dict((s, sum(r["state"] == s for r in rows)) for s in sorted({r["state"] for r in rows}))}')
    for r in rows:
        ci = r['confidence_95']
        print(f'{r["deck_id"][:8]} {r["bucket"]:10} HP {r["hp"]:3}/{r["max_hp"]:3} n={r["deck_size"]:2} '
              f'{r["state"]:17} wins {r["wins"]:2}/{r["completed"]:2} '
              f'{"" if not ci else f"[{100*ci[0]:.0f}–{100*ci[1]:.0f}%]":11} '
              f'probe {"-" if r["probe_seconds"] is None else round(r["probe_seconds"])}s')


def run(a):
    out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'controller.lock').open('a'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = dict(pool=str(a.pool.resolve()), pool_sha=sha(a.pool), worker_sha=sha(a.worker), workers=a.workers,
                  per_bucket=a.per_bucket, max_seconds=a.max_seconds, timeout=600, namespace=a.namespace,
                  sims=20000, probe=PROBE, total=TOTAL)
    cp = out / 'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('config changed on resume')
    cp.write_text(json.dumps(config, indent=2))
    worker = out / 'frozen/pv_worker'
    if not worker.exists():
        worker.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(a.worker, worker)
    if sha(worker) != config['worker_sha']:
        raise ValueError('frozen worker checksum mismatch')
    pool = pq.read_table(a.pool).to_pylist()
    lib_ids = {m['deck_id'] for m in json.loads((LIBRARY / 'manifest.json').read_text())}
    already = [r for r in pool if r['deck_id'] in lib_ids]
    picked = select([r for r in pool if r['deck_id'] not in lib_ids], already, a.per_bucket)
    used = {r['start']['seed'] for r in pool}
    used |= {r['start']['seed'] for r in pq.read_table(BENCH).to_pylist()}
    used |= {r['start']['seed'] for r in pq.read_table(LIBRARY / 'starts.parquet').to_pylist()}
    probe, fill, decks = [], [], []
    for r in picked:
        rows = generate(r, f'{a.namespace}:{r["deck_id"]}', 'screen', TOTAL, used)
        probe += rows[:PROBE]; fill += rows[PROBE:]
        st = r['start']
        decks.append(dict(deck_id=r['deck_id'], bucket=bucket(r), hp=st['hp'], max_hp=st['max_hp'],
                          deck_size=len(st['deck']), fight_ids=[x['fight_id'] for x in rows]))
    (out / 'decks.json').write_text(json.dumps(decks, indent=2))
    write_starts(probe, out / 'probe-starts.parquet')
    print(f'S0: {len(decks)} decks; probe {len(probe)} fights', flush=True)
    play(out, 'probe', out / 'probe-starts.parquet', None, worker, 20000, a.workers, teacher=True)
    ok = {r['deck_id'] for r in catalogue(out, decks, a.max_seconds) if r['state'] == 'filling'}
    survivors = [r for r in fill if r['deck_id'] in ok]
    write_starts(survivors, out / 'fill-starts.parquet')
    print(f'S0: {len(ok)}/{len(decks)} decks pass probe; fill {len(survivors)} fights', flush=True)
    play(out, 'fill', out / 'fill-starts.parquet', None, worker, 20000, a.workers, teacher=True)
    rows = catalogue(out, decks, a.max_seconds)
    (out / 'catalogue.json').write_text(json.dumps(rows, indent=2))
    (out / 'summary.json').write_text(json.dumps(dict(decks=len(rows), states={s: sum(r['state'] == s for r in rows)
                                                      for s in {r['state'] for r in rows}}), indent=2))
    print('S0 done', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--pool', type=Path, required=True)
    r.add_argument('--worker', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--workers', type=int, default=10)
    r.add_argument('--per-bucket', type=int, default=8)
    r.add_argument('--max-seconds', type=float, default=60)
    r.add_argument('--namespace', default='hdc-v1')
    s = sub.add_parser('status'); s.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'status':
        status(a.run)
    else:
        if not 1 <= a.workers <= 10:
            p.error('1–10 workers')
        run(a)


if __name__ == '__main__':
    main()
