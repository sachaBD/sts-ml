#!/usr/bin/env python3
"""Play fights from a parquet of starts with the PV agent or the guided-rollout teacher; write combat_v4 tables.

  play.py --starts STARTS.parquet --agent pv --model M.onnx --sims N [--explore] --workers 10 --out OUT
  play.py --starts STARTS.parquet --agent teacher --sims N --workers 10 --out OUT

STARTS: parquet with fight_id and start (combat_v4 fights layout; other columns are ignored). N `pv_worker` processes
each take one fight at a time (JSON lines over pipes, transport only). Results stream into
OUT/fights-<k>.parquet and OUT/search-<k>.parquet (a new part k every 256 fights; runs/schema=combat_v4/schema.py; the
agent string records the settings), OUT/decisions_stats-<k>.parquet (one row per searched decision: seconds, simulations, tree depth in actions and in player
turns crossed (visit-weighted mean and max over simulations), node count; depth is null for the teacher) and
OUT/summary.json. A fight that hits the turn/action cap is counted as `capped` and not written (it is not a loss).
"""
import argparse
import importlib.util
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'build/pv/agents/combat/pv/pv_worker'
spec = importlib.util.spec_from_file_location('combat_v4_schema', ROOT / 'runs/schema=combat_v4/schema.py')
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)

STATS = pa.schema([
    ('fight_id', pa.string()), ('step', pa.int16()), ('seconds', pa.float64()), ('simulations', pa.uint32()),
    ('mean_depth', pa.float64()), ('max_depth', pa.int32()), ('mean_turns', pa.float64()), ('max_turns', pa.int32()),
    ('nodes', pa.int64()), ('turns_hist', pa.list_(pa.int64()))])
TURN_STATS = pa.schema([
    ('fight_id', pa.string()), ('step', pa.int16()), ('turn', pa.int32()), ('seconds', pa.float64()),
    ('expansions', pa.int64()), ('network_calls', pa.int64()), ('evaluated_states', pa.int64()),
    ('root_children', pa.int64()), ('max_depth', pa.int32()), ('mean_leaf_depth', pa.float64()),
    ('fallback', pa.bool_()), ('reason', pa.string()), ('time_overshoot', pa.bool_()), ('reused', pa.bool_())])
PART = 256  # fights per output part file; each part is complete and readable as soon as it is written


def starts(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=['fight_id', 'start']):
        yield from batch.to_pylist()


def serve(command, source, lock, results):
    """One worker process: send a fight, read its result line, until the starts run out."""
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
            while True:
                with lock:
                    fight = next(source, None)
                if fight is None: break
                proc.stdin.write(json.dumps(fight) + '\n'); proc.stdin.flush()
                line = proc.stdout.readline()
                if not line: raise RuntimeError(f'worker died on {fight["fight_id"]} (exit {proc.wait()})')
                results.put(json.loads(line))
            proc.stdin.close()
    except BaseException as e:
        results.put(e)
    results.put(None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--starts', type=Path, required=True)
    p.add_argument('--agent', choices=['pv', 'teacher'], required=True)
    p.add_argument('--model', type=Path)
    p.add_argument('--sims', type=int, required=True)
    p.add_argument('--explore', action='store_true')
    p.add_argument('--oracle', action='store_true')
    p.add_argument('--turn-search', action='store_true', help='evaluation-only whole-turn oracle search; --sims is E')
    p.add_argument('--policy-only', action='store_true')
    p.add_argument('--sample-turns', action='store_true', help='sample moves proportional to visits in player turns 1–2')
    p.add_argument('--rollout-mix', type=float, help='weight of the guided-rollout leaf value in [0, 1] (worker default 0: network only)')
    p.add_argument('--c', type=float, help='PUCT exploration on min-max normalized Q (worker default when omitted)')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--worker', type=Path, default=WORKER)
    a = p.parse_args()
    if a.turn_search and (a.agent != 'pv' or not a.oracle or a.explore or a.policy_only or a.sample_turns or a.rollout_mix):
        p.error('--turn-search requires --agent pv --oracle and evaluation-only settings')
    if a.agent == 'pv':
        if not a.model: p.error('--agent pv needs --model')
        command = [str(a.worker), 'play', str(a.model), str(a.sims)] + ['--explore'] * a.explore + (['--c', str(a.c)] if a.c is not None else []) \
            + (['--rollout-mix', str(a.rollout_mix)] if a.rollout_mix is not None else []) \
            + ['--oracle'] * a.oracle + ['--turn-search'] * a.turn_search + ['--policy-only'] * a.policy_only + ['--sample-turns'] * a.sample_turns
    else:
        if a.model or a.explore or a.oracle or a.policy_only or a.sample_turns or a.c is not None or a.rollout_mix is not None: p.error('--model, --explore, --c and --rollout-mix are for --agent pv')
        command = [str(a.worker), 'teacher', str(a.sims)]
    a.out.mkdir(parents=True, exist_ok=True)

    source, lock, results = starts(a.starts), threading.Lock(), queue.Queue(maxsize=4 * a.workers)
    threads = [threading.Thread(target=serve, args=(command, source, lock, results), daemon=True) for _ in range(a.workers)]
    begin = time.time()
    for t in threads: t.start()
    tables = {'fights': (v4.FIGHTS, []), 'search': (v4.SEARCH, []), 'decisions_stats': (STATS, [])}
    if a.turn_search: tables['turn_stats'] = (TURN_STATS, [])
    part = 0

    def flush():
        nonlocal part
        if not tables['fights'][1]: return
        for name, (schema, rows) in tables.items():
            pq.write_table(pa.Table.from_pylist(rows, schema), a.out / f'{name}-{part}.parquet', compression='zstd')
            rows.clear()
        part += 1

    summary = dict(fights=0, wins=0, capped=0, decision_seconds=0.0)
    alive = len(threads)
    while alive:
        item = results.get()
        if item is None: alive -= 1; continue
        if isinstance(item, BaseException): raise item
        summary['decision_seconds'] += sum(s['seconds'] for s in item['turn_stats' if a.turn_search else 'stats'])
        if item['status'] == 'capped': summary['capped'] += 1; continue
        summary['fights'] += 1; summary['wins'] += item['fight']['won']
        tables['fights'][1].append(item['fight'])
        tables['search'][1].extend(item['search'])
        tables['decisions_stats'][1].extend(item['stats'])
        if a.turn_search: tables['turn_stats'][1].extend(item['turn_stats'])
        if len(tables['fights'][1]) == PART: flush()
    flush()
    played = summary['fights'] + summary['capped']
    summary |= dict(wall_seconds=time.time() - begin, workers=a.workers, command=command[1:],
                    mean_seconds_per_fight=summary['decision_seconds'] / played if played else None)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__': main()
