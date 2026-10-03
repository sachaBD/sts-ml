#!/usr/bin/env python3
"""Read / rebuild timings for one combat_v4 run (median of --repeats, seconds).

  bench_read.py RUN_OUT REPLAY_BIN [--repeats 5]
"""
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def timed(fn, repeats):
    times, result = [], None
    for _ in range(repeats):
        t = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), min(times), max(times), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path)
    ap.add_argument('replay')
    ap.add_argument('--repeats', type=int, default=5)
    a = ap.parse_args()
    fights_path, search_path = a.out / 'fights-00000.parquet', a.out / 'search-00000.parquet'
    db = duckdb.connect()
    db.execute(f"create view fights as select * from read_parquet('{fights_path}')")
    db.execute(f"create view search as select * from read_parquet('{search_path}')")
    results = {}

    def bench(name, fn):
        med, lo, hi, out = timed(fn, a.repeats)
        results[name] = {'median': med, 'min': lo, 'max': hi}
        return out

    bench('pyarrow read fights (all columns)', lambda: pq.read_table(fights_path))
    bench('pyarrow read search (all columns)', lambda: pq.read_table(search_path))
    fights_rows = bench('pyarrow fights -> Python dicts', lambda: pq.read_table(fights_path).to_pylist())
    bench('pyarrow search -> Python dicts', lambda: pq.read_table(search_path).to_pylist())
    bench('duckdb: win rate by encounter', lambda: db.sql(
        "select start.encounter, avg(won::int) from fights group by 1").fetchall())
    bench('duckdb: unnest all search children', lambda: db.sql(
        "select count(*), avg(c.visits) from search, unnest(children) as u(c)").fetchall())
    bench('duckdb: join fights x search on (fight_id, step), chosen move visit share', lambda: db.sql("""
        select avg(c.visits / s.simulations) from search s join fights f using (fight_id),
               unnest(s.children) as u(c) where c.action = f.actions[s.step + 1]""").fetchall())

    lines = '\n'.join(json.dumps({k: r[k] for k in ('fight_id', 'start', 'actions', 'won', 'final_hp')})
                      for r in fights_rows) + '\n'

    def replay():
        done = subprocess.run([a.replay], input=lines, capture_output=True, text=True, check=True)
        return json.loads(done.stdout)
    rep = bench('C++ rebuild + check all fights (process total, 1 thread)', replay)
    results['C++ rebuild detail (last run)'] = rep
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
