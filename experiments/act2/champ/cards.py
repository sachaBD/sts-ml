"""P(win | card in deck) for Champ fights. Usage: .venv/bin/python experiments/act2/champ/cards.py"""
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

rows = pq.read_table(Path(__file__).parent / "champ_fights.parquet").to_pylist()
seen, fights = set(), []
for r in rows:  # ah-dev-hyb == ah-dev-inc through Act 2; drop exact duplicates (same seed, deck, relics, hp)
    key = (r["seed"], tuple(r["deck"]), tuple(r["relics"]), r["hp"])
    if key not in seen:
        seen.add(key), fights.append(r)
n, base = len(fights), sum(r["won"] for r in fights) / len(fights)
print(f"{len(rows)} fights, {n} after dedup; win {base:.3f}; mean hp {sum(r['hp'] for r in fights) / n:.1f}")


def line(name, sel):
    k = len(sel)
    if not k:
        return
    p = sum(r["won"] for r in sel) / k
    print(f"  {name:28s} n={k:4d}  win={p:.2f} ± {math.sqrt(p * (1 - p) / k):.2f}  "
          f"hp={sum(r['hp'] for r in sel) / k:5.1f}  deck={sum(len(r['deck']) for r in sel) / k:4.1f}")


base_name = lambda c: c.rstrip("+")
counts = Counter(c for r in fights for c in {base_name(c) for c in r["deck"]})
print("\nP(win | card in deck), cards in >= 25 decks, sorted by win rate:")
for c in sorted((c for c, k in counts.items() if k >= 25), key=lambda c: -sum(
        r["won"] for r in fights if c in map(base_name, r["deck"])) / counts[c]):
    line(c, [r for r in fights if c in map(base_name, r["deck"])])

print("\nBy HP at fight start:")
for lo, hi in [(0, 40), (40, 55), (55, 70), (70, 200)]:
    line(f"hp {lo}-{hi}", [r for r in fights if lo <= r["hp"] < hi])
print("\nBy deck size:")
for lo, hi in [(0, 20), (20, 25), (25, 30), (30, 99)]:
    line(f"deck {lo}-{hi}", [r for r in fights if lo <= len(r["deck"]) < hi])
