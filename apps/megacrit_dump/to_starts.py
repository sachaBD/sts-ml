"""Champ starts (megacrit_champ_v1) -> pv_starts_v1 starts parquet for apps/pv/play.py.

  PYTHONPATH=. .venv/bin/python -m runs.run pv_starts_v1 <id> --no-compact --input <champ run> -- \
      .venv/bin/python apps/megacrit_dump/to_starts.py --champ-starts <champ run>/out/champ_starts.parquet \
      --runs <megacrit_runs_v1 run dirs> --out {out}

Writes out/starts.parquet (all converted decks) and out/starts_pv.parquet (same without Runic Dome: the PV agent rejects
hidden intents), plus summary.json. One row per (deck, copy k): copy 1 uses the human's seed_played, copies 2.. use
fresh seeds --seed0 + i. Everything not rebuilt from the human record (gold, RNG counters, room fields) is copied from
one row of a template starts file. Known approximations: potions empty; card misc 0; Searing Blow with >1 upgrade
is stored as upgraded=True; deck and relic order are sorted by name (true order is unknown); bottled = none.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps.megacrit_dump.champ_starts import header_array  # noqa: E402
from apps.pv.starts import start_type  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = ROOT / "runs/schema=pv_starts_v1/id=champ-bench2k/starts.parquet"
RUNIC_DOME = "runic_dome"
EMPTY_POTION = 1  # sts::Potion::EMPTY_POTION_SLOT
BASE_OK = {"event_card_upgrade_unknown", "bottled_relic"}
MISC_CARDS = {"ritual_dagger", "genetic_algorithm"}  # carry a per-card `misc` value we do not know
COPY_KEYS = ("ascension", "act", "floor", "encounter", "cur_room", "last_room", "burning_elite_buff", "gold",
             "misc_rng", "potion_rng")


def enum_index():
    cards = {n.lower(): i for i, n in enumerate(header_array("Cards.h", "cardEnumStrings"))}
    relics = {n.lower(): i for i, n in enumerate(header_array("Relics.h", "relicEnumNames"))}
    return cards, relics


def to_i64(seed) -> int | None:
    try:
        v = int(seed)
    except (TypeError, ValueError):
        return None
    return v if -(2 ** 63) <= v < 2 ** 63 else None


def to_u64(v: int) -> int:
    return v & (2 ** 64 - 1)  # two's complement


def convert(row: dict, seed: int, template: dict, cards: dict, relics: dict, stats=None):
    """One human champ-start row -> sts start struct, or None if it cannot be represented (reason in stats)."""
    stats = stats if stats is not None else collections.Counter()
    if row["unmapped"]:
        stats["skipped_unmapped"] += 1
        return None
    deck = []
    for d in row["deck"]:
        if d["card"] not in cards:
            stats["skipped_unmapped"] += 1
            return None
        if d["card"] == "searing_blow" and d["upgrades"] > 1:
            stats["searing_blow_multi_upgrade_clamped"] += 1
        if d["card"] in MISC_CARDS:
            stats["misc_card_in_deck"] += 1
        deck.append({"id": cards[d["card"]], "upgraded": d["upgrades"] > 0, "misc": 0})
    rel = []
    for r in row["relics"]:
        if r == "neows_lament":  # used up by Act 2
            stats["neows_lament_dropped"] += 1
            continue
        if r not in relics:
            stats["skipped_unmapped"] += 1
            return None
        rel.append({"id": relics[r], "data": 0})
    if row["hp"] is None or row["max_hp"] is None:
        stats["skipped_no_hp"] += 1
        return None
    cap = 3 if "potion_belt" in row["relics"] else 2
    start = {k: template[k] for k in COPY_KEYS}
    start.update(seed=to_u64(seed), hp=row["hp"], max_hp=row["max_hp"], potion_capacity=cap,
                 potions=[EMPTY_POTION] * cap, relics=rel, deck=deck, bottled=[-1, -1, -1])
    return start


def human_seeds(run_dirs, play_ids):
    """play_id -> seed_played (signed int64 or None) from megacrit_runs_v1 parts."""
    parts = []
    for d in map(Path, run_dirs):
        d = d / "out" / "runs" if (d / "out" / "runs").is_dir() else d / "runs" if (d / "runs").is_dir() else d
        parts += [str(p) for p in sorted(d.glob("part-*.parquet"))]
    db = duckdb.connect()
    db.execute("set memory_limit='1GB'; set threads=2")
    db.register("ids", pa.table({"play_id": sorted(play_ids)}))
    rows = db.execute("select play_id, any_value(seed_played) from read_parquet($1) where play_id in "
                      "(select play_id from ids) group by 1", [parts]).fetchall()
    return {pid: to_i64(s) for pid, s in rows}


def load_template(path) -> dict:
    return next(pq.ParquetFile(path).iter_batches(batch_size=1, columns=["start"])).to_pylist()[0]["start"]


def build(champ_rows, seeds, template, k, seed0, include="base-exact"):
    cards, relics = enum_index()
    stats = collections.Counter()
    out, i, seen = [], 0, set()
    for r in sorted(champ_rows, key=lambda r: r["play_id"]):
        if r["play_id"] in seen:
            stats["duplicate_play_id"] += 1
            continue
        seen.add(r["play_id"])
        reasons = {x.split(":")[0] for x in r["issues"]}
        base_exact = reasons <= BASE_OK
        if not (r["exact"] or (include == "base-exact" and base_exact)):
            stats["excluded_not_exact"] += 1
            continue
        hs = seeds.get(r["play_id"])
        if hs is None:
            stats["no_human_seed"] += 1
        first = convert(r, 0, template, cards, relics, stats)
        if first is None:
            continue
        stats["decks"] += 1
        stats["exact_decks"] += bool(r["exact"])
        for c in range(1, k + 1):
            if c == 1 and hs is not None:
                seed = hs
            else:  # fresh seed (also copy 1 if the human seed is unknown)
                seed = seed0 + i
                i += 1
            out.append({"fight_id": f"mc:{r['play_id']}:{c}", "start": dict(first, seed=to_u64(seed)),
                        "augment": "", "source_fight_id": r["play_id"], "play_id": r["play_id"],
                        "exact": bool(r["exact"]), "base_exact": base_exact, "copy": c})
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--champ-starts", nargs="+", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="megacrit_runs_v1 run dirs (for seed_played)")
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=997_000_000_000)
    ap.add_argument("--include", choices=["exact", "base-exact"], default="base-exact")
    ap.add_argument("--limit", type=int, default=None, help="keep only the first N fights (smoke tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in a.champ_starts:
        rows += pq.read_table(p).to_pylist()
    seeds = human_seeds(a.runs, {r["play_id"] for r in rows})
    fights, stats = build(rows, seeds, load_template(a.template), a.k, a.seed0, a.include)
    if a.limit:
        fights = fights[: a.limit]
    schema = pa.schema([("fight_id", pa.string()), ("start", start_type()), ("augment", pa.string()),
                        ("source_fight_id", pa.string()), ("play_id", pa.string()), ("exact", pa.bool_()),
                        ("base_exact", pa.bool_()), ("copy", pa.int32())])
    dome = int(header_array("Relics.h", "relicEnumNames").index("RUNIC_DOME"))
    pv = [f for f in fights if not any(r["id"] == dome for r in f["start"]["relics"])]
    for name, fs in (("starts.parquet", fights), ("starts_pv.parquet", pv)):
        pq.write_table(pa.Table.from_pylist(fs, schema=schema), out / name, compression="zstd")
    summary = {"champ_rows_in": len(rows), **stats, "fights": len(fights), "fights_pv_no_dome": len(pv),
               "dome_fights_dropped_for_pv": len(fights) - len(pv),
               "dome_decks": (len(fights) - len(pv)) // max(1, a.k), "k": a.k, "seed0": a.seed0, "include": a.include}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
