#!/usr/bin/env python3
"""Resample stored fights of combat_v3 bootstrap runs: same deck, relics and potions, new starting HP and fight RNG.

Spec: slop_docs/apps/fight_resample.md. One worker call per source fight plays all its samples; its rows go to
out/part-<source_episode_id>.parquet.
"""
import argparse
import json
import logging
import random
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
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sts_combat_rl.run import run_dir, run_parquet
from sts_combat_rl.schemas.combat_v3 import COMBAT_V3, NAME

log = logging.getLogger(__name__)
BINARY = Path("build-resample/fight_resample_worker")  # built by job.sh
MAX_SAMPLES = 1000  # episode_id = source_episode_id * 1000 + k
RUN_KEYS = {"id", "inputs", "encounter", "samples", "hp_sd", "random_potions", "value_run", "fights", "workers"}
COLUMNS = ["run_seed", "fight_index", "episode_id", "decision_index", "chosen_action", "ascension", "encounter", "floor",
           "starting_hp", "starting_max_hp"]


def inputs(run):
    """The source runs (each a bootstrap run: combat_v3 with no inputs), then the value run if any."""
    for source in run["inputs"]:
        meta = json.loads((run_dir(source) / "run.json").read_text())
        if meta["schema"] != NAME or meta.get("inputs"):
            sys.exit(f"{source}: not a bootstrap run (combat_v3 with no inputs); only those replay")
    return run["inputs"] + ([run["value_run"]] if "value_run" in run else [])


def load_fights(sources, encounter):
    """Source fights of `encounter`, by episode_id: (worker input without samples, stored decision_index 0 row)."""
    actions, starts, seen = defaultdict(list), {}, {}
    for source in sources:
        table = ds.dataset(run_parquet(source), format="parquet").to_table(
            columns=COLUMNS, filter=ds.field("row_kind") == "decision")
        for row in sorted(table.to_pylist(), key=lambda r: (r["run_seed"], r["fight_index"], r["decision_index"])):
            if seen.setdefault(row["run_seed"], source) != source:
                sys.exit(f"run_seed {row['run_seed']} is in both {seen[row['run_seed']]} and {source}")
            actions[row["run_seed"], row["fight_index"]].append(row["chosen_action"])
            if row["decision_index"] == 0 and row["encounter"] == encounter:
                starts[row["episode_id"]] = row
    return {episode: ({"run_seed": start["run_seed"], "ascension": start["ascension"],
                       "fight_index": start["fight_index"],
                       "actions": [actions[start["run_seed"], i] for i in range(start["fight_index"])]}, start)
            for episode, start in sorted(starts.items())}


def samples(start, count, hp_sd):
    """Sample k: episode_id = source * 1000 + k, starting HP ~ Normal(stored HP, hp_sd) rounded into [1, max HP]."""
    result = []
    for k in range(count):
        episode_id = start["episode_id"] * MAX_SAMPLES + k
        hp = round(random.Random(episode_id).gauss(start["starting_hp"], hp_sd))
        result.append({"episode_id": episode_id, "starting_hp": min(max(hp, 1), start["starting_max_hp"])})
    return result


def play(fight, start, weights, out):
    """One source fight -> out/part-<source_episode_id>.parquet; (teacher settings, per-sample results)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "fight.json").write_text(json.dumps(fight))
        began = time.monotonic()
        subprocess.run([str(BINARY), str(weights) if weights else "--no-weights", str(tmp / "fight.json"), str(tmp)],
                       check=True)
        seconds = time.monotonic() - began
        with (tmp / "fight.msgpack").open("rb") as source:
            result = msgpack.unpack(source, raw=False)
    table = pa.Table.from_pylist(result["rows"], schema=COMBAT_V3)
    firsts = [r for r in result["rows"] if r["row_kind"] == "decision" and r["decision_index"] == 0]
    wanted = [(s["episode_id"], s["starting_hp"]) for s in fight["samples"]]
    if [(r["episode_id"], r["starting_hp"]) for r in firsts] != wanted or any(
            r[c] != start[c] for r in firsts for c in ("encounter", "floor", "starting_max_hp")):
        raise RuntimeError(f"episode {start['episode_id']}: replayed fights don't match the stored fight / samples")
    pq.write_table(table, out / f"part-{start['episode_id']}.parquet", compression="zstd")
    return result["teacher"], [{"starting_hp": r["starting_hp"], "won": r["won"], "final_hp": r["final_hp"],
                                "terminal_value": r["terminal_value"], "seconds": seconds / len(firsts)}
                               for r in firsts]


def main(config_path, out):
    run = tomllib.loads(config_path.read_text())["run"]
    if unknown := sorted(set(run) - RUN_KEYS):
        sys.exit(f"unknown [run] keys: {unknown}")
    if not 1 <= run["samples"] <= MAX_SAMPLES:
        sys.exit(f"samples must be in [1, {MAX_SAMPLES}]")
    inputs(run)
    weights = None
    if "value_run" in run:
        value = json.loads((run_dir(run["value_run"]) / "run.json").read_text())["summary"]
        weights = run_dir(run["value_run"]) / "out" / value["weights"]
    fights = load_fights(run["inputs"], run["encounter"])
    fights = dict(list(fights.items())[:run.get("fights", len(fights))])
    random_potions = run.get("random_potions", False)
    for fight, start in fights.values():
        fight.update(random_potions=random_potions, samples=samples(start, run["samples"], run["hp_sd"]))
    log.info("%d %s source fights x %d samples, hp_sd %s, random_potions %s, teacher %s, %d workers -> %s",
             len(fights), run["encounter"], run["samples"], run["hp_sd"], random_potions,
             f"value_net {weights}" if weights else "guided_rollout", run["workers"], out)
    started, done, teacher = time.monotonic(), [], None
    with ThreadPoolExecutor(run["workers"]) as pool:
        futures = {pool.submit(play, fight, start, weights, out): episode for episode, (fight, start) in fights.items()}
        try:
            for i, future in enumerate(as_completed(futures), 1):
                teacher, results = future.result()
                done += results
                log.info("[%d/%d] episode %d: %s | total wins %d/%d", i, len(fights), futures[future],
                         " ".join(f"hp {r['starting_hp']}->{r['final_hp'] if r['won'] else 'lost'}" for r in results),
                         sum(r["won"] for r in done), len(done))
        except BaseException:
            log.exception("stopping: a fight failed; waiting for running workers")
            pool.shutdown(cancel_futures=True)
            raise
    summary = {"schema": NAME, "inputs": run["inputs"], "value_run": run.get("value_run"), "teacher": teacher,
               "encounter": run["encounter"], "samples": run["samples"], "hp_sd": run["hp_sd"],
               "random_potions": random_potions, "source_fights": len(fights), "fights": len(done),
               "wins": sum(r["won"] for r in done),
               "mean_terminal_value": sum(r["terminal_value"] for r in done) / len(done),
               "seconds_per_fight": sum(r["seconds"] for r in done) / len(done)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("done in %.1f min: %s", (time.monotonic() - started) / 60, json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, help="run output dir")
    parser.add_argument("--inputs", action="store_true", help="print the input run ids (for run.sh)")
    args = parser.parse_args()
    if args.inputs:
        print(*inputs(tomllib.loads(args.config.read_text())["run"]), sep="\n")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
        main(args.config, args.out)
