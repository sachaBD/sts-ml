"""Reproduce saved seed-0 p_win for natural rows through the GUI's scoring path (Ensemble with one checkpoint)."""
import json
from collections import Counter

from model import CHECKPOINTS, NATURAL, Ensemble, load, tables

t = tables()
for kind in ("cards", "relics", "potions"):
    print(kind, len(t[kind]))

rows, _ = load(NATURAL)
key = lambda r: (r["run_seed"], r["encounter"], r["start_hp"] if "start_hp" in r else r["starting_hp"], r["final_hp"])
counts = Counter(key(r) for r in rows)
by = {key(r): r for r in rows if counts[key(r)] == 1}
# ids in the data must match the header tables
names = {c["id"]: c["key"] for c in t["cards"]}
rel = {x["id"]: x["key"] for x in t["relics"]}
pot = {x["id"]: x["key"] for x in t["potions"]}
for r in rows:
    assert all(names[c["card_id"]] == c["name"] for c in r["pre"]["deck"])
    assert all(rel[x["relic_id"]] == x["name"] for x in r["pre"]["relics"])
    assert all(pot[x["potion_id"]] == x["name"] for x in r["pre"]["potions"])
print("id tables match all", len(rows), "natural rows")

preds = CHECKPOINTS[0].parent / "augmented-predictions.jsonl"
saved = [p for p in map(json.loads, preds.open()) if p["domain"] == "natural" and key(p) in by][:200]
ens = Ensemble(CHECKPOINTS[:1])
got = ens.score([by[key(p)] for p in saved])
err = max(abs(g["p_win"] - p["p_win"]) for g, p in zip(got, saved))
herr = max(abs(g["hp_if_win"] - p["mean_hp_if_win"]) for g, p in zip(got, saved))
print(f"{len(saved)} rows: max |dp_win| = {err:.2e}, max |d mean_hp_if_win| = {herr:.2e}")
assert err < 1e-5
