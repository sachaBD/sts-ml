#!/usr/bin/env python3
"""Audit that the reserved 600 final Demon Form seeds were never played/encoded/selected. Run from repo root.
Scans (a) every text artifact in the repo (excluding .git/.venv/build) for the final fight-id prefix or any final seed,
(b) every parquet file modified after the reservation (2026-10-06 09:08) for fight_id / seed / start.seed overlap."""
import json, os, sys, time
from pathlib import Path
import pyarrow.parquet as pq

REPO = Path.cwd(); assert (REPO / 'AGENTS.md').exists()
FINAL = REPO / 'runs/schema=combat_v4/date=2026-10-06/id=single-deck-demon-form-v1/out/final-reserved.parquet'
SKIP = {'.git', '.venv', 'build', 'node_modules', '__pycache__'}
CUTOFF = time.mktime(time.strptime('2026-10-06 09:08:00', '%Y-%m-%d %H:%M:%S'))
rows = pq.read_table(FINAL).to_pylist()
ids = {r['fight_id'] for r in rows}; seeds = {r['start']['seed'] for r in rows}
seed_strs = {str(s) for s in seeds}
assert len(ids) == len(seeds) == 600
hits = {'text': [], 'parquet': []}; scanned = {'text': 0, 'parquet': 0, 'parquet_errors': 0}
for root, dirs, files in os.walk(REPO):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for f in files:
        p = Path(root) / f
        if p.resolve() == FINAL.resolve(): continue
        if f.endswith(('.jsonl', '.json', '.log', '.csv', '.md', '.txt')):
            try:
                if p.stat().st_size > 2_000_000_000: continue
                text = p.read_text(errors='ignore')
            except OSError: continue
            scanned['text'] += 1
            if 'demon-form-fresh-v1:final' in text or any(s in text for s in seed_strs):
                found = [s for s in seed_strs if s in text][:3]
                hits['text'].append({'path': str(p.relative_to(REPO)), 'final_prefix': 'demon-form-fresh-v1:final' in text, 'seeds': found})
        elif f.endswith('.parquet') and p.stat().st_mtime > CUTOFF:
            scanned['parquet'] += 1
            try:
                schema = pq.read_schema(p); names = set(schema.names); cols = [c for c in ('fight_id', 'seed', 'start') if c in names]
                if not cols: continue
                t = pq.read_table(p, columns=cols)
                hit = {}
                if 'fight_id' in cols: hit['ids'] = len(set(t['fight_id'].to_pylist()) & ids)
                if 'seed' in cols: hit['seed'] = len(set(t['seed'].to_pylist()) & seeds)
                if 'start' in cols: hit['start_seed'] = len({s['seed'] for s in t['start'].to_pylist() if s and 'seed' in s} & seeds)
                if any(hit.values()): hits['parquet'].append({'path': str(p.relative_to(REPO)), **hit})
            except Exception as e:
                scanned['parquet_errors'] += 1
print(json.dumps({'final': str(FINAL), 'final_ids': len(ids), 'scanned': scanned, 'hits': hits}, indent=1))
