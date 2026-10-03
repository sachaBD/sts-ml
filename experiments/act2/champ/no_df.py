"""Champ fights WITHOUT Demon Form: P(win | card), and strength-scaling counts. Usage: .venv/bin/python experiments/act2/champ/no_df.py"""
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

rows = pq.read_table(Path(__file__).parent / "champ_fights.parquet").to_pylist()
seen, fights = set(), []
for r in rows:
    key = (r["seed"], tuple(r["deck"]), tuple(r["relics"]), r["hp"])
    if key not in seen:
        seen.add(key), fights.append(r)
base = lambda c: c.rstrip("+")
fights = [r for r in fights if not any(base(c) == "demon_form" for c in r["deck"]) and r["hp"] >= 55]
p0 = sum(r["won"] for r in fights) / len(fights)
print(f"no Demon Form, hp >= 55: n={len(fights)} win={p0:.2f}")


def line(name, sel):
    k = len(sel)
    p = sum(r["won"] for r in sel) / k if k else float("nan")
    print(f"  {name:30s} n={k:4d}  win={p:.2f} ± {math.sqrt(p * (1 - p) / max(k, 1)):.2f}  hp={sum(r['hp'] for r in sel) / max(k, 1):5.1f}")


counts = Counter(c for r in fights for c in {base(c) for c in r["deck"]})
print("\nP(win | card), n >= 30, top 15 / bottom 8:")
order = sorted((c for c, k in counts.items() if k >= 30),
               key=lambda c: -sum(r["won"] for r in fights if c in map(base, r["deck"])) / counts[c])
for c in order[:15] + ["..."] + order[-8:]:
    if c == "...":
        print("  ...")
        continue
    line(c, [r for r in fights if c in map(base, r["deck"])])
STR = {"inflame", "spot_weakness", "limit_break"}
print("\nstrength scaling cards (inflame / spot_weakness / limit_break; copies):")
for k in range(4):
    line(f"{k}{'+' if k == 3 else ''}", [r for r in fights if (sum(base(c) in STR for c in r["deck"]) >= k if k == 3
                                                              else sum(base(c) in STR for c in r["deck"]) == k)])
print("\nrelics (n >= 40), top 8:")
rc = Counter(x for r in fights for x in r["relics"])
ro = sorted((x for x, k in rc.items() if k >= 40), key=lambda x: -sum(r["won"] for r in fights if x in r["relics"]) / rc[x])
for x in ro[:8]:
    line(x, [r for r in fights if x in r["relics"]])
