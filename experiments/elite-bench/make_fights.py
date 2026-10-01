#!/usr/bin/env python3
"""Benchmark fight list: 600 elite fights per encounter from buckets 0-1 of the scaled-search baseline
(run_seed % 10 in (0, 1); no v3 net trained on them), chosen by SHA-256 of episode_id, never by outcome.
Also writes the 200-per-elite subset (first 200 by the same order) used by the scaling stage.

    PYTHONPATH=python .venv/bin/python experiments/elite-bench/make_fights.py
"""
import csv
import hashlib
from pathlib import Path

from sts_combat_rl.query import connect

HERE = Path(__file__).parent
rows = connect().sql("""
    select episode_id, run_seed, encounter, floor, starting_hp from combat_v3
    where id = 'act1-all-bosses-a20-scaled-search' and source_episode_id is null and category = 'elite'
      and run_seed % 10 in (0, 1) and row_kind = 'decision' and decision_index = 0""").fetchall()
for name, n in (("bench_fights.csv", 600), ("bench_subset.csv", 200)):
    with (HERE / name).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("source_episode_id", "run_seed", "encounter", "floor", "starting_hp"))
        for enc in ("gremlin_nob", "lagavulin", "three_sentries"):
            fights = sorted((r for r in rows if r[2] == enc), key=lambda r: hashlib.sha256(str(r[0]).encode()).digest())
            assert len(fights) >= n, (enc, len(fights))
            w.writerows(fights[:n])
print("wrote bench_fights.csv (600/elite) and bench_subset.csv (200/elite)")
