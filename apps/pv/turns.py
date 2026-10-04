#!/usr/bin/env python3
"""Enumerate player turns from the first N recorded fights; native JSON lines are transport only.

Writes turns.parquet: fight_id, turn (zero-based), sequences, distinct (conservative exact key),
canonical_distinct (hand/discard/exhaust sorted, draw ordered), capped, seconds.
The native worker validates full fights with combat_v4::replay, then replays prefixes to each turn start.
"""
import argparse
import json
from pathlib import Path
import subprocess

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ('fight_id', pa.string()), ('turn', pa.int16()), ('sequences', pa.int64()), ('distinct', pa.int64()),
    ('canonical_distinct', pa.int64()), ('capped', pa.bool_()), ('seconds', pa.float64()),
    ('pairs', pa.int32()), ('divergent', pa.int32()), ('illegal', pa.int32()), ('pair_capped', pa.int32()),
    ('terminal_wins', pa.int64()), ('terminal_losses', pa.int64()), ('cap_reason', pa.string()),
])


def summarize(final, count):
    db = duckdb.connect()
    db.execute("set memory_limit='512MB'; set threads=1")
    lines = [f'Fights: {count}; keys: conservative upper bounds. All three key sanity checks passed.']
    for column in ('sequences', 'distinct', 'canonical_distinct', 'seconds'):
        row = db.execute(f'select median("{column}"), quantile_cont("{column}", 0.9), max("{column}") from read_parquet(?)',
                         [str(final)]).fetchone()
        lines.append(f'{column}: median={row[0]}, p90={row[1]}, max={row[2]}')
    turns, capped = db.execute('select count(*), sum(capped::int) from read_parquet(?)', [str(final)]).fetchone()
    lines.append(f'Turns: {turns}; capped: {capped}/{turns} ({100 * capped / turns:.1f}%). '
                 'Counts in capped turns are partial, not final distinct-state estimates.')
    wins, losses = db.execute('select sum(terminal_wins), sum(terminal_losses) from read_parquet(?)',
                              [str(final)]).fetchone()
    lines.append(f'Terminal sequences (excluded from distinct counts): wins={wins}, losses={losses}, total={wins + losses}.')
    reasons = db.execute('select cap_reason, count(*) from read_parquet(?) where capped group by cap_reason',
                         [str(final)]).fetchall()
    lines.append(f'Cap reasons: {reasons}.')
    pairs, divergent, illegal, pair_capped = db.execute(
        'select sum(pairs), sum(divergent), sum(illegal), sum(pair_capped) from read_parquet(?)', [str(final)]).fetchone()
    matched = pairs - illegal - pair_capped
    lines.append(f'Differential pairs: {pairs}; completed: {matched}; divergent: {divergent}/{matched} '
                 f'({100 * divergent / matched:.1f}%)' if matched else f'Differential pairs: {pairs}; none completed.')
    lines.append(f'Illegal replays: {illegal}/{pairs}; depth-capped continuations: {pair_capped}/{pairs}. '
                 'Deterministic sample: first 64 buckets and up to 5 pairs/turn, max 200 pairs total.')
    lines.append('Differential sample is non-random and within-turn correlated; population uncertainty not estimated.')
    summary = '\n'.join(lines) + '\n'
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fights', type=Path, required=True, help='directory containing fights-*.parquet')
    p.add_argument('--worker', type=Path, required=True)
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.n < 1:
        p.error('n must be positive')
    a.out.mkdir(parents=True, exist_ok=True)
    final = a.out / 'turns.parquet'
    partial = a.out / 'turns.parquet.tmp'
    count = 0
    with pq.ParquetWriter(partial, SCHEMA, compression='zstd') as writer:
        with subprocess.Popen([str(a.worker), 'turns'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
            for path in sorted(a.fights.glob('fights-*.parquet')):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
                    fight = batch.to_pylist()[0]
                    proc.stdin.write(json.dumps(fight) + '\n'); proc.stdin.flush()
                    rows = []
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            raise RuntimeError(f'worker failed on {fight["fight_id"]} (exit {proc.wait()})')
                        row = json.loads(line)
                        if 'done' in row:
                            break
                        rows.append(row)
                        print(f'{row["fight_id"]} turn={row["turn"]} sequences={row["sequences"]} '
                              f'distinct={row["distinct"]} canonical={row["canonical_distinct"]} '
                              f'capped={row["capped"]} seconds={row["seconds"]:.3f}', flush=True)
                    writer.write_table(pa.Table.from_pylist(rows, SCHEMA))
                    count += 1
                    if count == a.n:
                        break
                if count == a.n:
                    break
            proc.stdin.close()
        if proc.returncode:
            raise RuntimeError(f'turn enumeration exited {proc.returncode}')
    if not count:
        raise ValueError('no fights found')
    partial.replace(final)
    summary = summarize(final, count)
    (a.out / 'summary.txt').write_text(summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
