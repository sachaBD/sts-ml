#!/usr/bin/env python3
"""Natural combat_transition_v1 rows from the fights of overworld_v1 runs (multi-act; e.g. experiments/act2).

  PYTHONPATH=. .venv/bin/python -m runs.run combat_transition_v1 ID --input overworld_v1/DATE/RUN ... -- \
      .venv/bin/python apps/combat_transition/from_overworld.py overworld_v1/DATE/RUN ... --out {out}

overworld_v1 fight steps hold only the state AFTER exitBattle plus hp_before. The pre-fight state is rebuilt from the
latest earlier recorded state: a fight's post state, a card-reward state plus the picked card, or a decision's
after-state for the chosen option. Out-of-combat steps SimpleAgent plays (chests, undecided events, ...) are not
recorded, so a rebuilt pre state can be stale. replay = 'ok' only when it checks out against the fight: HP equals
hp_before, and the deck and relic ids equal the post state's (a fight leaves both unchanged apart from rare in-fight
additions such as Parasite); otherwise replay = 'stale' (learn.load drops those rows).

final_hp (the combat_outcome endpoint: in-battle HP before exitBattle) is the post-state HP minus the post-victory heal
(Burning Blood 6, Black Blood 12, Meat on the Bone 12 when at or below half HP), which is exact unless the heal was
capped at max HP; then the midpoint of the possible range is used and hp_capped = true.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from runs.run import RUNS

PRE_KEYS = ("deck", "floor", "gold", "hp", "max_hp", "potion_capacity", "potions", "relics")
HEAL = {"burning_blood": 6, "black_blood": 12}


def battle_final_hp(post, won):
    """(in-battle final HP, heal was capped) from the post-exitBattle state."""
    if not won:
        return 0, False
    hp, mx = post["hp"], post["max_hp"]
    names = {r["name"] for r in post["relics"]}
    heal = sum(v for k, v in HEAL.items() if k in names)
    if heal == 0 and "meat_on_the_bone" not in names:
        return hp, False
    if hp < mx:
        base = hp - heal
        # Meat on the Bone checks in-battle HP <= maxHp / 2 (integer division) after the other heals.
        if "meat_on_the_bone" in names and base - 12 >= 1 and base - 12 <= mx // 2:
            base -= 12
        return max(base, 1), False
    lo = max(mx - heal - (12 if "meat_on_the_bone" in names else 0), 1)
    return (lo + mx) // 2, True


def ids(xs, key):
    return sorted(x[key] for x in xs)


def fights(run, source):
    """Yield one row per fight of one overworld_v1 run record."""
    last, act = None, 1
    for step in run["steps"]:
        kind = step["kind"]
        if kind == "start":
            last = step["state"]
        elif kind == "pick":
            last = copy.deepcopy(step["state"])
            choice = step.get("choice")
            if isinstance(choice, int) and 0 <= choice < len(step["options"]):
                last["deck"].append(dict(step["options"][choice]))
        elif kind == "decide":
            after = step.get("after")  # the chosen option's after-state (absent for event lookahead decisions)
            if isinstance(after, dict) and "deck" in after:
                last = after
        elif kind == "fight":
            post = step["state"]
            act = post.get("act", act)
            pre = {k: copy.deepcopy(last[k]) for k in PRE_KEYS} if last else None
            ok = pre is not None and pre["hp"] == step["hp_before"] \
                and ids(pre["deck"], "card_id") == ids(post["deck"], "card_id") \
                and ids(pre["relics"], "relic_id") == ids(post["relics"], "relic_id")
            if pre is not None:
                pre["hp"] = step["hp_before"]
                pre["floor"] = post["floor"]
            final_hp, capped = battle_final_hp(post, step["won"])
            seed = run["seed"]
            yield dict(source_run_id=source, run_seed=seed, bucket=seed % 10, act=act, floor=post["floor"],
                       encounter=step["encounter"], category=step["category"], boss=step.get("boss"),
                       starting_hp=step["hp_before"], starting_max_hp=(pre or post)["max_hp"],
                       won=bool(step["won"]), final_hp=final_hp, battle_final_hp=final_hp, hp_capped=capped,
                       any_random=False, oracle=False, replay="ok" if ok else "stale",
                       pre=pre, post={k: post[k] for k in PRE_KEYS})
            last = post


def run_path(run_id):
    schema, date, rid = run_id.split("/")
    return RUNS / f"schema={schema}" / f"date={date}" / f"id={rid}" / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="overworld_v1/DATE/ID runs")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect()
    rows, counts = [], Counter()
    for source in a.sources:
        files = f"{run_path(source)}/runs-*.parquet"
        for (record,) in db.sql(f"select record_json from read_parquet('{files}')").fetchall():
            for row in fights(json.loads(record), source):
                counts[(source, row["replay"])] += 1
                rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows), a.out / "part-000.parquet")
    summary = dict(sources=a.sources, fights=len(rows),
                   replay=Counter(r["replay"] for r in rows),
                   hp_capped=sum(r["hp_capped"] for r in rows),
                   per_source={f"{s}:{k}": v for (s, k), v in sorted(counts.items())},
                   act2_boss_ok=Counter(r["encounter"] for r in rows
                                        if r["act"] == 2 and r["category"] == "boss" and r["replay"] == "ok"))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
