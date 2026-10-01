#!/usr/bin/env python3
"""Frozen fight lists for the easy-pool cheap-agent study: 250 fights per easy encounter, chosen by SHA-256 of
episode_id (never by outcome) from act1-all-bosses-a20-scaled-search.

  fights.csv   buckets 0-1 (run_seed % 10 in (0, 1)): dev set, used to pick the cheap agent
  confirm.csv  buckets 2-3: fresh fights, played once to confirm the pick

easy_number = which easy fight of its run this is (1 = floor 1; 2, 3 = the later easy fights), by floor.

    PYTHONPATH=. .venv/bin/python experiments/act-1-combat/2026-09-29-easy-cheap-agent/make_fights.py
"""
import csv
import hashlib
from pathlib import Path

from runs.query import connect

HERE = Path(__file__).parent
N = 250
ENCOUNTERS = ("cultist", "jaw_worm", "small_slimes", "two_louse")
COLS = ("source_episode_id", "run_seed", "encounter", "floor", "easy_number", "starting_hp", "starting_max_hp")

rows = connect().sql("""
    select episode_id, run_seed, encounter, floor,
           row_number() over (partition by run_seed order by floor) easy_number, starting_hp, starting_max_hp
    from combat_v3
    where id = 'act1-all-bosses-a20-scaled-search' and source_episode_id is null and category = 'easy'
      and row_kind = 'decision' and decision_index = 0""").fetchall()
for name, buckets in (("fights.csv", (0, 1)), ("confirm.csv", (2, 3))):
    with (HERE / name).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for enc in ENCOUNTERS:
            pool = sorted((r for r in rows if r[2] == enc and r[1] % 10 in buckets),
                          key=lambda r: hashlib.sha256(str(r[0]).encode()).digest())
            assert len(pool) >= N, (enc, len(pool))
            w.writerows(pool[:N])
    print(f"wrote {name}: {N}/encounter, buckets {buckets}")
