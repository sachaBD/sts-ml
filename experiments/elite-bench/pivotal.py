#!/usr/bin/env python3
"""Pivotal fight list (after S1-S3): fights where some fair 20k run (MCTS or v3, any salt) lost, and the fight is
winnable (the oracle won it, or some fair run won it). Written in the bench_fights.csv format.

    PYTHONPATH=. .venv/bin/python experiments/elite-bench/pivotal.py [--prefix bench] [--store runs]
"""
import argparse
import csv
from pathlib import Path

from bench import ELITES, arms, load

HERE = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--prefix", default="bench")
p.add_argument("--store", default="runs")
p.add_argument("--out", default=str(HERE / "pivotal_fights.csv"))
a = p.parse_args()

A = arms(load(a.prefix, a.store))
oracle = A["oracle-20k"][0]
fair = [A[arm][s] for arm in ("mcts-20k", "v3-20k") for s in sorted(A.get(arm, {})) if s < 4]
print(f"fair 20k runs used: {len(fair)} (expect 8)")
common = set(oracle).intersection(*fair)
rows = []
for ep in sorted(common):
    wins = [r[ep]["won"] for r in fair]
    if not all(wins) and (oracle[ep]["won"] or any(wins)):
        f = oracle[ep]
        rows.append((ep, f["seed"], f["enc"], "", f["hp"]))
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(("source_episode_id", "run_seed", "encounter", "floor", "starting_hp"))
    w.writerows(rows)
counts = {e: sum(r[2] == e for r in rows) for e in ELITES}
print(f"{len(rows)} pivotal of {len(common)} common fights {counts} -> {a.out}")
