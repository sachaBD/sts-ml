"""Demon Form decks vs The Champ: split by upgrade / HP, and print decks to read by hand.
Usage: .venv/bin/python experiments/act2/champ/demon_form.py"""
import math
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

rows = pq.read_table(Path(__file__).parent / "champ_fights.parquet").to_pylist()
seen, fights = set(), []
for r in rows:
    key = (r["seed"], tuple(r["deck"]), tuple(r["relics"]), r["hp"])
    if key not in seen:
        seen.add(key), fights.append(r)
df = lambda r: [c for c in r["deck"] if c.startswith("demon_form")]


def line(name, sel):
    k = len(sel)
    p = sum(r["won"] for r in sel) / k if k else float("nan")
    print(f"  {name:30s} n={k:4d}  win={p:.2f} ± {math.sqrt(p * (1 - p) / max(k, 1)):.2f}  "
          f"hp={sum(r['hp'] for r in sel) / max(k, 1):5.1f}")


print("Demon Form split:")
line("no demon form", [r for r in fights if not df(r)])
line("demon_form (unupgraded only)", [r for r in fights if df(r) and "demon_form+" not in df(r)])
line("demon_form+", [r for r in fights if "demon_form+" in df(r)])
line("2+ copies", [r for r in fights if len(df(r)) >= 2])
print("\nBy HP, with / without Demon Form:")
for lo, hi in [(0, 40), (40, 55), (55, 70), (70, 200)]:
    line(f"hp {lo}-{hi} no DF", [r for r in fights if lo <= r["hp"] < hi and not df(r)])
    line(f"hp {lo}-{hi} DF", [r for r in fights if lo <= r["hp"] < hi and df(r)])
print("\nDemon Form prevalence by run (policy):")
for run in sorted({r["run"] for r in fights}):
    sel = [r for r in fights if r["run"] == run]
    print(f"  {run:22s} n={len(sel):4d} DF share={sum(bool(df(r)) for r in sel) / len(sel):.2f}")


def show(r):
    c = Counter(r["deck"])
    deck = ", ".join(f"{k}{'×' + str(v) if v > 1 else ''}" for k, v in sorted(c.items()))
    print(f"\n[{'WIN ' if r['won'] else 'LOSS'}] hp {r['hp']}/{r['max_hp']} → {r['final_hp']}  {r['run']} seed {r['seed']}  "
          f"({len(r['deck'])} cards, {r['decisions']} decisions)\n  deck: {deck}\n  relics: {', '.join(r['relics'])}"
          f"\n  potions: {', '.join(r['potions']) or '-'}")


random.seed(0)
dfw = [r for r in fights if df(r) and r["won"]]
dfl = [r for r in fights if df(r) and not r["won"]]
print(f"\n===== Demon Form LOSSES with hp >= 55 (all {sum(r['hp'] >= 55 for r in dfl)} of {len(dfl)} losses)")
for r in sorted((r for r in dfl if r["hp"] >= 55), key=lambda r: -r["hp"]):
    show(r)
print(f"\n===== Demon Form WINS (8 random of {len(dfw)})")
for r in random.sample(dfw, 8):
    show(r)
