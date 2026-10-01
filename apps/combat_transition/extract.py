#!/usr/bin/env python3
"""Combat transitions (schema combat_transition_v1) of stored combat_v3 bootstrap runs.

Each source run seed is replayed from its seed with the stored chosen actions (no search; apps/combat_transition/
worker.cpp), and one row is written per fight: the persistent state right before the fight (pre) and right after
exitBattle, before rewards (post), with the stored outcome and the fight's teacher settings.

Only bootstrap runs (no inputs): their fights are the played trajectory of a whole act 1 run. A fight whose replay no
longer reproduces the stored fight (simulator drift) is written with replay = 'diverged' and no states; the rest of
that run is not replayed (not written). Counts in summary.json.

out/part-000.parquet, one row per fight; summary.json.
"""
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from apps.common.app import check_keys, main, required, run_json, snapshot, write_json
from apps.common.worker import run_parallel, run_worker
from sts_combat_rl.run import RUNS

log = logging.getLogger(__name__)
SCHEMA = "combat_transition_v1"
BUILT = "build/main/combat_transition_worker"
RUN_KEYS = {"id", "sources", "workers", "limit"}


def inputs(config):
    return list(config["run"]["sources"])


def run_dir(run_id):
    schema, date, rid = run_id.split("/")
    return RUNS / f"schema={schema}" / f"date={date}" / f"id={rid}"


def load_fights(run_id):
    """run_seed -> [fight dict in fight_index order] with the stored actions and outcome of each fight."""
    db = duckdb.connect()
    rows = db.sql(f"""
        select run_seed, fight_index, any_value(episode_id) episode_id, any_value(encounter) encounter,
               any_value(category) category, any_value(floor) floor, any_value(ascension) ascension,
               any_value(starting_hp) starting_hp, any_value(starting_max_hp) starting_max_hp,
               any_value(won) won, any_value(final_hp) final_hp, any_value(potions) potions,
               bool_or(coalesce(was_random, false)) any_random, bool_or(coalesce(oracle, false)) oracle,
               count(*) decisions, list(chosen_action order by decision_index) actions,
               any_value(global_numeric) filter (where decision_index = 0) start_global_numeric,
               any_value(cards) filter (where decision_index = 0) start_cards,
               any_value(monsters) filter (where decision_index = 0) start_monsters,
               list(decision_index order by decision_index) idx
        from read_parquet('{run_dir(run_id)}/out/*.parquet', union_by_name = true)
        where row_kind = 'decision' group by all order by run_seed, fight_index""").fetchall()
    cols = ["run_seed", "fight_index", "episode_id", "encounter", "category", "floor", "ascension", "starting_hp",
            "starting_max_hp", "won", "final_hp", "potions", "any_random", "oracle", "decisions", "actions",
            "start_global_numeric", "start_cards", "start_monsters", "idx"]
    runs = defaultdict(list)
    for r in rows:
        f = dict(zip(cols, r))
        if f.pop("idx") != list(range(f["decisions"])):
            raise RuntimeError(f"{run_id}: fight {f['episode_id']} has missing or repeated decisions")
        runs[f["run_seed"]].append(f)
    for seed, fights in runs.items():
        if [f["fight_index"] for f in fights] != list(range(len(fights))):
            raise RuntimeError(f"{run_id}: run seed {seed} is missing fights")
    return runs


def extract(config, config_path, out):
    run = config["run"]
    check_keys(run, RUN_KEYS, "run")
    sources = required(run, "sources", "run", list)
    workers = required(run, "workers", "run", int)
    binary, worker_sha256 = snapshot(Path(BUILT), out)
    rows, counts = [], Counter()
    for source in sources:
        meta = run_json(source)
        if meta["inputs"]:
            sys.exit(f"{source}: not a bootstrap run (it has inputs)")
        teacher = (meta.get("summary") or {}).get("teacher")
        runs = load_fights(source)
        seeds = sorted(runs)[: run["limit"]] if "limit" in run else sorted(runs)
        log.info("%s: %d run seeds, %d fights, status %s, teacher %s", source, len(seeds),
                 sum(len(runs[s]) for s in seeds), meta["status"], teacher)

        def request(seed):
            fights = runs[seed]
            return {"run_seed": seed, "ascension": fights[0]["ascension"],
                    "fights": [{"actions": f["actions"], "won": f["won"], "final_hp": f["final_hp"],
                                "potions": f["potions"], "encounter": f["encounter"]} for f in fights]}

        def on_result(seed, result):
            stored = runs[seed]
            got = {f["fight_index"]: f for f in result["fights"]}
            for f in stored:
                r = got.get(f["fight_index"])
                status = r["replay"] if r else "not_replayed"
                counts[status] += 1
                budget = None
                if teacher:
                    by = teacher.get("simulations_by_category")
                    budget = by.get(f["category"]) if by else teacher.get("simulations")
                check = None  # check_start (apps/common/replay.py START): differing fields, [] = identical
                if r and "start" in r:
                    check = [c for c in ("encounter", "floor", "starting_hp", "starting_max_hp") if r[c] != f[c]]
                    check += [c for c in ("global_numeric", "cards", "monsters") if r["start"][c] != f["start_" + c]]
                rows.append({
                    "source_run_id": source, "run_seed": seed, "bucket": seed % 10, "episode_id": f["episode_id"],
                    "fight_index": f["fight_index"], "floor": f["floor"], "encounter": f["encounter"],
                    "category": f["category"], "ascension": f["ascension"], "starting_hp": f["starting_hp"],
                    "starting_max_hp": f["starting_max_hp"], "won": f["won"], "final_hp": f["final_hp"],
                    "potions": f["potions"], "any_random": f["any_random"], "oracle": f["oracle"],
                    "decisions": f["decisions"], "teacher": json.dumps(teacher, sort_keys=True) if teacher else None,
                    "simulations": budget, "replay": status, "start_check": check, "reason": r.get("reason") if r else None,
                    "battle_final_hp": r.get("battle_final_hp") if r else None,
                    "escaped": r.get("escaped") if r else None,
                    "max_hp_changed": (r["post"]["max_hp"] != r["pre"]["max_hp"]) if r and "post" in r else None,
                    "pre": r.get("pre") if r else None, "post": r.get("post") if r else None})

        run_parallel(lambda seed: run_worker(binary, request(seed)), seeds, workers, on_result)
    rows.sort(key=lambda r: (r["source_run_id"], r["run_seed"], r["fight_index"]))
    pq.write_table(pa.Table.from_pylist(rows), out / "part-000.parquet", compression="zstd")
    reasons = Counter(r["reason"] for r in rows if r["replay"] == "diverged")
    checks = Counter((r["replay"], ",".join(r["start_check"]) or "identical") for r in rows if r["start_check"] is not None)
    write_json(out / "summary.json", {"schema": SCHEMA, "sources": sources, "worker_sha256": worker_sha256,
                                      "fights": len(rows), "replay": dict(counts), "diverged_reasons": dict(reasons),
                                      "start_check": {f"{a}: {b}": n for (a, b), n in sorted(checks.items())}})
    log.info("done: %d fights, replay %s, start_check %s", len(rows), dict(counts), dict(checks))


if __name__ == "__main__":
    main(extract, inputs)
