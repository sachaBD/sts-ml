#!/usr/bin/env python3
"""DAgger collection (slop_docs/apps/dagger.md): a frozen value-net learner replays eligible training fights
while the guided-rollout teacher labels every reached decision; completed fights -> combat_v3 parquet.

Does not train. Parquet metadata: schema=combat_v3, collection_method=dagger, training_target=teacher_root_only.
"""
import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import msgpack
import pyarrow as pa
import pyarrow.parquet as pq
from sts_combat_rl.run import bootstrap_inputs, run_dir
from sts_combat_rl.schemas.combat_v3 import COMBAT_V3, NAME as COMBAT_V3_NAME

log = logging.getLogger(__name__)
BUILT = Path("build-dagger/dagger_worker")  # built by job.sh
METADATA = {"collection_method": "dagger", "training_target": "teacher_root_only"}
# The replayed decision 0 must match the stored one on these columns.
START = ("episode_id", "encounter", "floor", "starting_hp", "starting_max_hp", "global_numeric", "cards", "monsters")
COLLECTION_KEYS = {"count", "seed", "episodes", "exclude_run_seeds", "max_decisions", "max_turns", "timeout_seconds"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_json(run_id):
    return json.loads((run_dir(run_id) / "run.json").read_text())


def source_runs(learner):
    """The learner's bootstrap combat_v3 input runs (no inputs of their own)."""
    return bootstrap_inputs(learner)


def select(collection, checkpoint, available):
    """Episodes to play: training episodes of the checkpoint minus exclusions, whose run seed is still in
    the sources (`available`; independent of outcomes); explicit list or seeded sample."""
    if unknown := sorted(set(collection) - COLLECTION_KEYS):
        sys.exit(f"unknown [collection] keys: {unknown}")
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


def index_sources(sources, seeds):
    """run_seed -> (source, part) for the given seeds found in the sources' parts."""
    where = {}
    for source in sources:
        for part in sorted((run_dir(source) / "out").glob("*.parquet")):
            try:
                found = set(pq.read_table(part, columns=["run_seed"])["run_seed"].to_pylist()) & seeds
            except Exception:  # unreadable: still being written; fine unless it holds a selected seed
                log.warning("skipping unreadable part %s", part)
                continue
            for seed in found:
                if seed in where:
                    raise RuntimeError(f"run seed {seed} in both {where[seed][1]} and {part}")
                where[seed] = (source, part)
    return where


def load_fights(where, episodes):
    """episode_id -> worker input + stored decision 0 row, from the source parts holding those run seeds.

    Only the parts that hold a selected seed are read in full (a live source run may be writing others);
    each used part is hashed before and after reading, so a part changing underneath fails the run.
    """
    seeds = {e // 100 for e in episodes}
    if missing := sorted(seeds - set(where)):
        raise RuntimeError(f"run seeds missing from the sources: {missing[:10]}")
    parts, fights = {}, {}
    for source, part in sorted(set(where.values())):
        before = sha256(part)
        rows = pq.read_table(part, columns=["run_seed", "fight_index", "decision_index", "row_kind", "chosen_action",
                                            "ascension", *START]).to_pylist()
        if sha256(part) != before:
            raise RuntimeError(f"{part} changed while reading")
        parts[str(part)] = {"source": source, "sha256": before}
        actions = defaultdict(list)  # (run_seed, fight_index) -> decision rows in play order
        for row in rows:
            if row["row_kind"] == "decision" and row["run_seed"] in seeds:
                actions[row["run_seed"], row["fight_index"]].append(row)
        for episode in (e for e in episodes if where[e // 100][1] == part):
            seed, index = divmod(episode, 100)
            earlier = []
            for i in range(index):
                decisions = sorted(actions.get((seed, i), []), key=lambda r: r["decision_index"])
                if not decisions or [r["decision_index"] for r in decisions] != list(range(len(decisions))):
                    raise RuntimeError(f"episode {episode}: fight {i} decisions missing in {part}")
                earlier.append([r["chosen_action"] for r in decisions])
            start = [r for r in actions.get((seed, index), []) if r["decision_index"] == 0]
            if len(start) != 1:
                raise RuntimeError(f"episode {episode}: no stored decision 0 in {part}")
            fights[episode] = ({"run_seed": seed, "ascension": start[0]["ascension"], "fight_index": index,
                                "actions": earlier}, start[0], source)
    return fights, parts


def snapshot(path, out, name):
    """Read-only copy in out/; its sha256."""
    copy = out / name
    shutil.copy2(path, copy)
    copy.chmod(0o444 if name.endswith(".bin") else 0o555)
    return copy.resolve(), sha256(copy)


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def play(episode, fight, start, binary, weights, out, max_decisions, max_turns, timeout):
    """One fight; completed -> out/part-<episode_id>.parquet (atomic). Returns its status record."""
    record = {"episode": episode, "status": "failed"}
    began = time.monotonic()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "fight.json").write_text(json.dumps({**fight, "max_decisions": max_decisions, "max_turns": max_turns}))
        try:
            subprocess.run([str(binary), str(weights), str(tmp / "fight.json"), str(tmp)], check=True,
                           timeout=timeout, capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            return {**record, "status": "timeout", "seconds": time.monotonic() - began}
        except subprocess.CalledProcessError as error:
            return {**record, "error": error.stderr[-2000:], "seconds": time.monotonic() - began}
        with (tmp / "fight.msgpack").open("rb") as source:
            result = msgpack.unpack(source, raw=False)
    record.update(seconds=time.monotonic() - began, decisions=result["decisions"],
                  learner=result["learner"], teacher=result["teacher"])
    diagnostics = result["diagnostics"]
    record["learner_simulations"] = sum(d["learner_simulations"] for d in diagnostics)
    record["teacher_simulations"] = sum(d["teacher_simulations"] for d in diagnostics)
    record["teacher_disagreements"] = sum(d["learner_action"] != d["teacher_action"] for d in diagnostics)
    write_json(out / "diagnostics" / f"{episode}.json", diagnostics)
    if differs := [c for c in START if result["start"][c] != start[c]]:
        return {**record, "error": f"start state differs from the source on {differs}"}
    if result["status"] != "completed":
        return {**record, "status": result["status"]}
    rows = result["rows"]
    table = pa.Table.from_pylist(rows, schema=COMBAT_V3.with_metadata({"schema": COMBAT_V3_NAME, **METADATA}))
    tmp = out / f".part-{episode:08d}.tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out / f"part-{episode:08d}.parquet")
    return {**record, "status": "completed", "rows": len(rows), "won": rows[0]["won"],
            "terminal_value": rows[0]["terminal_value"]}


def main(config_path, out):
    config = tomllib.loads(config_path.read_text())
    run, collection = config["run"], config.get("collection", {})
    learner = run["input"]
    if (meta := run_json(learner)).get("status") != "done":
        sys.exit(f"{learner}: status {meta.get('status')}, need done")
    learner_out = run_dir(learner) / "out"
    checkpoint = json.loads((learner_out / meta["summary"]["checkpoint_json"]).read_text())
    sources = source_runs(learner)
    where = index_sources(sources, set(checkpoint["train_run_seeds"]))
    missing = len(set(checkpoint["train_run_seeds"]) - set(where))
    if missing:
        log.warning("%d checkpoint training run seeds are no longer in the sources; not eligible", missing)
    episodes, eligible = select(collection, checkpoint, set(where))
    max_decisions = collection.get("max_decisions", 500)
    max_turns = collection.get("max_turns", 50)
    timeout = collection.get("timeout_seconds", 600)
    fights, parts = load_fights(where, episodes)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    weights, weights_sha = snapshot(learner_out / meta["summary"]["weights"], out, "value_weights.bin")
    binary, binary_sha = snapshot(BUILT, out, "dagger_worker")
    shutil.copy2(config_path, out / "config.toml")
    manifest = {
        **METADATA, "schema": COMBAT_V3_NAME, "exploration": False,
        "teacher_root_value": "visit-weighted average of explored root edge values (sum valueSum / root visits)",
        "rows": "decision rows only; actions/root_value/simulations_used = teacher search; chosen_action = "
                "learner's executed move; outcome columns = learner's finished fight",
        "learner_run": learner, "weights_sha256": weights_sha,
        "checkpoint_sha256": sha256(learner_out / meta["summary"]["checkpoint"]),
        "checkpoint_json_sha256": sha256(learner_out / meta["summary"]["checkpoint_json"]),
        "sources": sources, "source_parts": parts, "config": config, "worker_sha256": binary_sha,
        "max_decisions": max_decisions, "max_turns": max_turns, "timeout_seconds": timeout, "eligible_episodes": eligible,
        "training_run_seeds_missing_from_sources": missing,
        "episodes": episodes, "run_seeds": sorted({e // 100 for e in episodes}),
    }
    (out / "inputs").mkdir(exist_ok=True)
    for episode, (fight, start, source) in fights.items():  # exactly what each worker replays and is checked against
        write_json(out / "inputs" / f"{episode}.json", {"source": source, "request": fight, "expected_start": start})
    write_json(out / "collection.json", manifest)
    log.info("DAgger: %d of %d eligible episodes from %s, learner %s, %d workers -> %s",
             len(episodes), eligible, ", ".join(sources), learner, run["workers"], out)
    records, started = [], time.monotonic()
    with ThreadPoolExecutor(run["workers"]) as pool:
        futures = [pool.submit(play, e, f, s, binary, weights, out, max_decisions, max_turns, timeout)
                   for e, (f, s, _) in fights.items()]
        for future in as_completed(futures):
            records.append(r := future.result())
            log.info("[%d/%d] episode %d %s%s %.0fs", len(records), len(fights), r["episode"], r["status"],
                     f" won={r['won']} tv={r['terminal_value']:.3f} disagree={r['teacher_disagreements']}/{r['decisions']}"
                     if r["status"] == "completed" else f" {r.get('error', '')}", r["seconds"])
    counts = {s: sum(r["status"] == s for r in records) for s in ("completed", "capped", "turn_limit", "timeout", "failed")}
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--inputs", action="store_true", help="print the learner run and source run ids (for run.sh)")
    args = parser.parse_args()
    if args.inputs:
        learner = tomllib.loads(args.config.read_text())["run"]["input"]
        print(learner, *source_runs(learner), sep="\n")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
        sys.exit(main(args.config, args.out))
