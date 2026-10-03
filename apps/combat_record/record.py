#!/usr/bin/env python3
"""Record the first fights of seeded Ironclad A20 act 1 runs into combat_v4 (runs/schema=combat_v4/schema.py).

  record.py --out OUT --worker BIN --seed0 S --seeds N --max-fights K --sims SIMS --search-stats on|off --id ID

Runs one apps/combat_record worker per seed in parallel (the worker plays, rebuilds and checks each fight and
prints its rows as JSON), then writes out/fights-00000.parquet and, with search stats on, out/search-00000.parquet,
validated against the combat_v4 schema. out/summary.json: counts, file sizes and where the time went.
"""
import argparse
import importlib.util
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = Path(__file__).resolve().parents[2] / 'runs/schema=combat_v4/schema.py'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--worker', required=True)
    ap.add_argument('--seed0', type=int, required=True)
    ap.add_argument('--seeds', type=int, required=True)
    ap.add_argument('--max-fights', type=int, required=True)
    ap.add_argument('--sims', type=int, required=True)
    ap.add_argument('--search-stats', choices=['on', 'off'], required=True)
    ap.add_argument('--id', required=True, help='collection name, prefix of fight_id')
    ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()

    def play(seed):
        argv = [a.worker, str(seed), str(a.max_fights), str(a.sims), a.id, a.search_stats]
        done = subprocess.run(argv, capture_output=True, text=True, check=True)
        timing = json.loads(done.stderr.strip().splitlines()[-1])['timing']
        return done.stdout, timing

    t0 = time.monotonic()
    with ThreadPoolExecutor(a.workers) as pool:
        outputs = list(pool.map(play, range(a.seed0, a.seed0 + a.seeds)))
    wall = {'play': time.monotonic() - t0}

    t = time.monotonic()
    rows = [json.loads(line) for stdout, _ in outputs for line in stdout.splitlines()]
    fights = [r['fight'] for r in rows]
    search = [s for r in rows for s in r['search']]
    wall['json_parse'] = time.monotonic() - t

    spec = importlib.util.spec_from_file_location('combat_v4_schema', SCHEMA)
    schema = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema)
    a.out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, table_rows in (('fights', fights), ('search', search)):
        if name == 'search' and a.search_stats == 'off':
            continue
        t = time.monotonic()
        table = pa.Table.from_pylist(table_rows, schema=schema.TABLES[name])
        wall[f'{name}_arrow_build'] = time.monotonic() - t
        t = time.monotonic()
        path = a.out / f'{name}-00000.parquet'
        pq.write_table(table, path, compression='zstd')
        wall[f'{name}_parquet_write'] = time.monotonic() - t
        files[name] = {'rows': table.num_rows, 'bytes': path.stat().st_size}

    worker = {k: sum(tm[k] for _, tm in outputs) for k in ('search', 'record', 'check')}
    (a.out / 'summary.json').write_text(json.dumps({
        'fights': len(fights), 'decisions': sum(len(f['actions']) for f in fights),
        'won': sum(f['won'] for f in fights), 'search_stats': a.search_stats, 'files': files,
        'worker_seconds': worker, 'wall_seconds': wall,
        'stdout_bytes': sum(len(stdout) for stdout, _ in outputs)}, indent=2))


if __name__ == '__main__':
    main()
