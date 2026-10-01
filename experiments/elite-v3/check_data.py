#!/usr/bin/env python3
"""Audit a v4 data run (fight_resample output) before training on it; exit 1 if a gate fails.

    PYTHONPATH=python .venv/bin/python experiments/elite-v3/check_data.py OUT_DIR [--min-fights N] [--max-skipped F]

OUT_DIR: the run's out/ directory (works for scratch runs, which the query views never read).
Gates: at least --min-fights fights; at most --max-skipped of the source fights skipped (replay diverged);
every row has the encoding v4 columns; decision rows carry legal actions whose tried moves have visits.
Also prints what training will see: fights per elite, rows, potion / relic coverage, legal moves per decision.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

POTIONS = {2: "Ambrosia", 3: "Ancient", 4: "Attack", 5: "Blessing of the Forge", 6: "Block", 7: "Blood",
           9: "Colorless", 10: "Cultist", 12: "Dexterity", 13: "Distilled Chaos", 14: "Duplication", 15: "Elixir",
           16: "Energy", 17: "Entropic Brew", 19: "Essence of Steel", 20: "Explosive", 21: "Fairy", 22: "Fear",
           23: "Fire", 24: "Flex", 26: "Fruit Juice", 27: "Gambler's Brew", 29: "Heart of Iron",
           30: "Liquid Bronze", 31: "Liquid Memories", 34: "Power", 35: "Regen", 36: "Skill", 38: "Snecko Oil",
           39: "Speed", 41: "Strength", 42: "Swift", 43: "Weak"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--min-fights", type=int, default=1)
    p.add_argument("--max-skipped", type=float, default=0.15)
    a = p.parse_args()
    parts = sorted(a.out.glob("*.parquet"))
    if not parts:
        sys.exit(f"no parquet in {a.out}")
    t = pa.concat_tables([pq.read_table(x) for x in parts], promote_options="default")
    failures = []
    decision = pc.equal(t["row_kind"], "decision")
    firsts = t.filter(pc.and_(decision, pc.equal(t["decision_index"], 0)))
    fights = firsts.num_rows
    print(f"parts {len(parts)}, rows {t.num_rows:,}, decision rows {pc.sum(decision).as_py():,}, fights {fights:,}")
    print("fights per encounter:", dict(Counter(firsts["encounter"].to_pylist())))
    print(f"wins {pc.sum(firsts['won']).as_py()} / {fights}")
    if fights < a.min_fights:
        failures.append(f"only {fights} fights (< {a.min_fights})")
    summary = a.out / "summary.json"
    if summary.exists():
        s = json.loads(summary.read_text())
        skipped, sources = s.get("skipped_diverged", 0), s.get("source_fights", fights)
        print(f"skipped (replay diverged): {skipped} of {sources} source fights; teacher {s.get('teacher')}")
        if sources and skipped / sources > a.max_skipped:
            failures.append(f"{skipped}/{sources} source fights skipped (> {a.max_skipped:.0%})")
    else:
        print("no summary.json (run unfinished?)")
    if "v4_encoding_version" not in t.column_names or t["v4_encoding_version"].null_count:
        failures.append("rows without the encoding v4 columns")
    decisions = t.filter(decision)
    with_actions = decisions["legal_actions"].null_count == 0
    if not with_actions:
        failures.append(f"{decisions['legal_actions'].null_count} decision rows without legal_actions")
    moves = pc.list_value_length(decisions["legal_actions"]).to_numpy(zero_copy_only=False)
    tried = pc.list_value_length(decisions["actions"]).to_numpy(zero_copy_only=False)
    print(f"legal moves per decision: mean {moves.mean():.1f}, max {moves.max()}; tried moves mean {tried.mean():.1f}")
    if (tried == 0).any():
        failures.append(f"{int((tried == 0).sum())} decision rows without visits")
    held = [x for x in firsts["potion_tokens"].to_pylist()]
    print(f"fights starting with >=1 potion: {sum(1 for x in held if x)} / {fights}")
    counts = Counter(POTIONS.get(tok["potion_id"], tok["potion_id"]) for x in held for tok in x)
    print("potions at fight start:", dict(counts.most_common()))
    relics = Counter(tok["relic_id"] - 1 for x in firsts["relic_tokens"].to_pylist() for tok in x)
    print(f"relic ids at fight start (RelicId): {len(relics)} distinct; most common {relics.most_common(12)}")
    if failures:
        print("GATE FAILED: " + "; ".join(failures))
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
