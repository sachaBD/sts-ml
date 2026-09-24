#!/usr/bin/env python3
"""DAgger collection (slop_docs/apps/dagger.md): a frozen value-net learner replays eligible training fights
while the guided-rollout teacher labels every reached decision; completed fights -> combat_v3 parquet.

Does not train. Parquet metadata: schema=combat_v3, collection_method=dagger, training_target=teacher_root_only.
"""
import logging
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from apps.common.app import check_keys, main, sha256, snapshot, value_run, write_json
from apps.common.replay import COLUMNS, START, check_start, decision_rows, replay_requests
from apps.common.worker import run_parallel, run_worker, write_part
from sts_combat_rl.run import bootstrap_inputs
from sts_combat_rl.schemas.combat_v3 import NAME as COMBAT_V3_NAME

log = logging.getLogger(__name__)
BUILT = Path("build/dagger/dagger_worker")  # built by apps/common/job.sh
METADATA = {"collection_method": "dagger", "training_target": "teacher_root_only"}
RUN_KEYS = {"id", "input", "workers"}
COLLECTION_KEYS = {"count", "seed", "episodes", "exclude_run_seeds", "max_decisions", "max_turns", "timeout_seconds"}
STATUSES = ("completed", "capped", "turn_limit", "timeout", "failed")


def inputs(config):
    """The learner run, then its bootstrap combat_v3 input runs (the sources of the replayed fights)."""
    learner = config["run"]["input"]
    return [learner, *bootstrap_inputs(learner)]


def select(collection, checkpoint, available):
    """Episodes to play: training episodes of the checkpoint minus exclusions, whose run seed is still in
    the sources (`available`; independent of outcomes); explicit list or seeded sample."""
    check_keys(collection, COLLECTION_KEYS, "collection")
    if ("episodes" in collection) == ("count" in collection):
        sys.exit("set exactly one of [collection] episodes / count")
    train_seeds = set(checkpoint["train_run_seeds"])
    forbidden_seeds = set(checkpoint["validation_run_seeds"]) | set(collection.get("exclude_run_seeds", []))
    eligible = sorted(e for e in set(checkpoint["train_episode_ids"]) - set(checkpoint["validation_episode_ids"])
                      if e // 100 in train_seeds and e // 100 not in forbidden_seeds and e // 100 in available)
    if "episodes" in collection:
        episodes = collection["episodes"]
        bad = sorted(set(episodes) - set(eligible))
        if bad or len(set(episodes)) != len(episodes) or not episodes:
            sys.exit(f"episodes must be distinct eligible training episodes; not eligible: {bad}")
        return episodes, len(eligible)
    count = collection["count"]
    if not 0 < count <= len(eligible):
        sys.exit(f"count {count}: {len(eligible)} eligible episodes")
    return sorted(random.Random(collection.get("seed", 0)).sample(eligible, count)), len(eligible)


def play(episode, request, start, binary, weights, out, timeout):
    """One fight; completed -> out/part-<episode_id>.parquet. Returns its status record."""
    record = {"episode": episode, "status": "failed"}
    began = time.monotonic()
    try:
        result = run_worker(binary, request, weights, timeout)
    except subprocess.TimeoutExpired:
        return {**record, "status": "timeout", "seconds": time.monotonic() - began}
    except RuntimeError as error:
        return {**record, "error": str(error), "seconds": time.monotonic() - began}
    record.update(seconds=time.monotonic() - began, decisions=result["decisions"],
                  learner=result["learner"], teacher=result["teacher"])
    diagnostics = result["diagnostics"]
    record["learner_simulations"] = sum(d["learner_simulations"] for d in diagnostics)
    record["teacher_simulations"] = sum(d["teacher_simulations"] for d in diagnostics)
    record["teacher_disagreements"] = sum(d["learner_action"] != d["teacher_action"] for d in diagnostics)
    write_json(out / "diagnostics" / f"{episode}.json", diagnostics)
    try:
        check_start(episode, result["start"], start, ("episode_id", *START))
    except RuntimeError as error:
        return {**record, "error": str(error)}
    if result["status"] != "completed":
        return {**record, "status": result["status"]}
    rows = result["rows"]
    write_part(out, episode, rows, {"schema": COMBAT_V3_NAME, **METADATA})
    return {**record, "status": "completed", "rows": len(rows), "won": rows[0]["won"],
            "terminal_value": rows[0]["terminal_value"]}


def collect(config, config_path, out):
    run, collection = config["run"], config.get("collection", {})
    check_keys(run, RUN_KEYS, "run")
    learner, *sources = inputs(config)
    value = value_run(learner)
    checkpoint = value.meta
    train_seeds = set(checkpoint["train_run_seeds"])
    available = {row["run_seed"] for row in decision_rows(sources, ["run_seed"], train_seeds)}
    if missing := len(train_seeds - available):
        log.warning("%d checkpoint training run seeds are no longer in the sources; not eligible", missing)
    episodes, eligible = select(collection, checkpoint, available)
    max_decisions = collection.get("max_decisions", 500)
    max_turns = collection.get("max_turns", 50)
    timeout = collection.get("timeout_seconds", 600)
    fights = replay_requests(decision_rows(sources, COLUMNS, {e // 100 for e in episodes}), episodes)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    weights, weights_sha = snapshot(value.weights, out, "value_weights.bin")
    binary, binary_sha = snapshot(BUILT, out)
    shutil.copy2(config_path, out / "config.toml")
    manifest = {
        **METADATA, "schema": COMBAT_V3_NAME, "exploration": False,
        "teacher_root_value": "visit-weighted average of explored root edge values (sum valueSum / root visits)",
        "rows": "decision rows only; actions/root_value/simulations_used = teacher search; chosen_action = "
                "learner's executed move; outcome columns = learner's finished fight",
        "learner_run": learner, "weights_sha256": weights_sha,
        "checkpoint_sha256": sha256(value.checkpoint), "checkpoint_json_sha256": sha256(value.checkpoint_json),
        "sources": sources, "config": config, "worker_sha256": binary_sha,
        "max_decisions": max_decisions, "max_turns": max_turns, "timeout_seconds": timeout, "eligible_episodes": eligible,
        "training_run_seeds_missing_from_sources": missing,
        "episodes": episodes, "run_seeds": sorted({e // 100 for e in episodes}),
    }
    (out / "inputs").mkdir(exist_ok=True)
    for episode, (request, start) in fights.items():  # exactly what each worker replays and is checked against
        request.update(max_decisions=max_decisions, max_turns=max_turns)
        write_json(out / "inputs" / f"{episode}.json", {"request": request, "expected_start": start})
    write_json(out / "collection.json", manifest)
    log.info("DAgger: %d of %d eligible episodes from %s, learner %s, %d workers -> %s",
             len(episodes), eligible, ", ".join(sources), learner, run["workers"], out)
    records, started = [], time.monotonic()

    def done(_, r):
        records.append(r)
        log.info("[%d/%d] episode %d %s%s %.0fs", len(records), len(fights), r["episode"], r["status"],
                 f" won={r['won']} tv={r['terminal_value']:.3f} disagree={r['teacher_disagreements']}/{r['decisions']}"
                 if r["status"] == "completed" else f" {r.get('error', '')}", r["seconds"])

    run_parallel(lambda item: play(item[0], *item[1], binary, weights, out, timeout), fights.items(), run["workers"], done)
    counts = {s: sum(r["status"] == s for r in records) for s in STATUSES}
    complete = counts["completed"] == len(episodes)
    summary = {**{k: manifest[k] for k in ("schema", *METADATA, "learner_run", "sources", "weights_sha256", "checkpoint_sha256",
                                           "worker_sha256", "max_decisions", "max_turns", "timeout_seconds")},
               "collection_status": "complete" if complete else "incomplete",
               "selected": len(episodes), "attempted": len(records), **counts,
               "missing": sorted(set(episodes) - {r["episode"] for r in records if r["status"] == "completed"}),
               "learner": next((r["learner"] for r in records if "learner" in r), None),
               "teacher": next((r["teacher"] for r in records if "teacher" in r), None),
               "wins": sum(r.get("won", False) for r in records),
               "fights": sorted(records, key=lambda r: r["episode"]),
               "minutes": (time.monotonic() - started) / 60}
    write_json(out / "summary.json", summary)
    log.info("done: %s %s in %.1f min", summary["collection_status"], counts, summary["minutes"])
    return 0 if complete else 1


if __name__ == "__main__":
    main(collect, inputs)
