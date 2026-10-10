#!/usr/bin/env python3
"""Parquet half of the final-seed audit: every parquet under runs/ modified after the reservation (2026-10-06 09:08),
fight_id / seed / start.seed overlap with the reserved 600. (Text half: grep -F over all text artifacts.)"""
import json, os, time
from pathlib import Path
import pyarrow.parquet as pq

REPO = Path.cwd(); assert (REPO / 'AGENTS.md').exists()
FINAL = REPO / 'runs/schema=combat_v4/date=2026-10-06/id=single-deck-demon-form-v1/out/final-reserved.parquet'
CUTOFF = time.mktime(time.strptime('2026-10-06 09:08:00', '%Y-%m-%d %H:%M:%S'))
rows = pq.read_table(FINAL).to_pylist(); ids = {r['fight_id'] for r in rows}; seeds = {r['start']['seed'] for r in rows}
hits, n, err, skipped = [], 0, [], 0
for p in sorted(REPO.glob('runs/**/*.parquet')):
    if p.resolve() == FINAL.resolve() or p.stat().st_mtime <= CUTOFF: continue
    n += 1
    try:
        names = set(pq.read_schema(p).names); cols = [c for c in ('fight_id', 'seed', 'start') if c in names]
        if not cols: skipped += 1; continue
        t = pq.read_table(p, columns=cols); h = {}
        if 'fight_id' in cols: h['ids'] = len(set(t['fight_id'].to_pylist()) & ids)
        if 'seed' in cols: h['seed'] = len(set(t['seed'].to_pylist()) & seeds)
        if 'start' in cols: h['start_seed'] = len({s.get('seed') for s in t['start'].to_pylist() if isinstance(s, dict)} & seeds)
        if any(h.values()): hits.append({'path': str(p.relative_to(REPO)), **h})
    except Exception as e:
        err.append([str(p.relative_to(REPO)), repr(e)[:200]])
print(json.dumps({'scanned_after_cutoff': n, 'no_id_columns': skipped, 'errors': err, 'hits': hits}, indent=1))
