#!/usr/bin/env python3
"""Resample stored fights of combat_v3 bootstrap runs: same deck, relics and potions, new starting HP and fight RNG.

Spec: slop_docs/apps/fight_resample.md. [run] query selects the fights (sts_combat_rl.query); they must come from
bootstrap runs, whose earlier fights rebuild each one. One worker call per source fight plays all its samples; its rows go to
out/part-<source_episode_id>.parquet.
"""
import logging
import random
import sys
import time
from pathlib import Path

from apps.common.app import check_keys, main, run_json, snapshot, value_run, write_json
from apps.common.replay import check_start, decision_rows, replay_requests
from sts_combat_rl import query
from apps.common.worker import run_parallel, run_worker, write_part
from sts_combat_rl.schemas.combat_v3 import NAME

log = logging.getLogger(__name__)
BUILT = Path("build/resample/fight_resample_worker")  # built by apps/common/job.sh
MAX_SAMPLES = 1000  # episode_id = source_episode_id * 1000 + k
RUN_KEYS = {"id", "query", "samples", "hp_sd", "random_potions", "value_run", "fights", "workers"}
COLUMNS = ["run_seed", "fight_index", "episode_id", "decision_index", "chosen_action", "ascension", "encounter", "floor",
           "starting_hp", "starting_max_hp"]


def inputs(config):
    """The source runs of the queried fights (each a bootstrap run: no inputs of its own), then the value run if any."""
    run = config["run"]
    sources = query.run_ids(run["query"])
    if not sources:
        sys.exit(f"no fights: {run['query']}")
    for source in sources:
        if run_json(source).get("inputs"):
            sys.exit(f"{source}: not a bootstrap run (it has inputs); only bootstrap fights replay")
    return sources + ([run["value_run"]] if "value_run" in run else [])


def samples(start, count, hp_sd):
    """Sample k: episode_id = source * 1000 + k, starting HP ~ Normal(stored HP, hp_sd) rounded into [1, max HP]."""
    result = []
    for k in range(count):
        episode_id = start["episode_id"] * MAX_SAMPLES + k
        hp = round(random.Random(episode_id).gauss(start["starting_hp"], hp_sd))
        result.append({"episode_id": episode_id, "starting_hp": min(max(hp, 1), start["starting_max_hp"])})
    return result


def play(episode, request, start, binary, weights, out):
    """One source fight -> out/part-<source_episode_id>.parquet; (teacher settings, per-sample results)."""
    began = time.monotonic()
    result = run_worker(binary, request, weights)
    seconds = time.monotonic() - began
    firsts = [r for r in result["rows"] if r["row_kind"] == "decision" and r["decision_index"] == 0]
    wanted = [(s["episode_id"], s["starting_hp"]) for s in request["samples"]]
    if [(r["episode_id"], r["starting_hp"]) for r in firsts] != wanted:
        raise RuntimeError(f"episode {episode}: replayed fights don't match the samples")
    for first in firsts:
        check_start(episode, first, start, ("encounter", "floor", "starting_max_hp"))
    write_part(out, episode, result["rows"])
    return result["teacher"], [{"starting_hp": r["starting_hp"], "won": r["won"], "final_hp": r["final_hp"],
                                "terminal_value": r["terminal_value"], "seconds": seconds / len(firsts)}
                               for r in firsts]


def resample(config, config_path, out):
    run = config["run"]
    check_keys(run, RUN_KEYS, "run")
    if not 1 <= run["samples"] <= MAX_SAMPLES:
        sys.exit(f"samples must be in [1, {MAX_SAMPLES}]")
    sources = [r for r in inputs(config) if r != run.get("value_run")]
    weights = None
    if "value_run" in run:
        weights, _ = snapshot(value_run(run["value_run"]).weights, out, "value_weights.bin")
    binary, worker_sha256 = snapshot(BUILT, out)
    episodes = sorted({r["episode_id"] for r in query.rows(run["query"], ["episode_id"])})
    episodes = episodes[:run.get("fights", len(episodes))]
    fights = replay_requests(decision_rows(sources, COLUMNS, {e // 100 for e in episodes}), episodes)
    random_potions = run.get("random_potions", False)
    for request, start in fights.values():
        request.update(random_potions=random_potions, samples=samples(start, run["samples"], run["hp_sd"]))
    log.info("%d source fights x %d samples, hp_sd %s, random_potions %s, teacher %s, %d workers -> %s",
             len(fights), run["samples"], run["hp_sd"], random_potions,
             f"value_net {weights}" if weights else "guided_rollout", run["workers"], out)
    started, done, teacher, finished = time.monotonic(), [], None, 0

    def on_result(item, result):
        nonlocal teacher, finished
        teacher, results = result
        done.extend(results)
        finished += 1
        log.info("[%d/%d] episode %d: %s | total wins %d/%d", finished, len(fights), item[0],
                 " ".join(f"hp {r['starting_hp']}->{r['final_hp'] if r['won'] else 'lost'}" for r in results),
                 sum(r["won"] for r in done), len(done))

    run_parallel(lambda item: play(item[0], *item[1], binary, weights, out), fights.items(), run["workers"], on_result)
    summary = {"schema": NAME, "query": run["query"], "sources": sources, "value_run": run.get("value_run"),
               "teacher": teacher, "worker_sha256": worker_sha256, "samples": run["samples"],
               "hp_sd": run["hp_sd"], "random_potions": random_potions, "source_fights": len(fights), "fights": len(done),
               "wins": sum(r["won"] for r in done),
               "mean_terminal_value": sum(r["terminal_value"] for r in done) / len(done),
               "seconds_per_fight": sum(r["seconds"] for r in done) / len(done)}
    write_json(out / "summary.json", summary)
    log.info("done in %.1f min: %s", (time.monotonic() - started) / 60, summary)


if __name__ == "__main__":
    main(resample, inputs)
