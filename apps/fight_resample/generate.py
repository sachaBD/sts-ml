#!/usr/bin/env python3
"""Resample stored fights of combat_v3 bootstrap runs: same deck, relics and potions, new starting HP and fight RNG.

[run] query selects the fights (runs.query); they must come from
bootstrap runs, whose earlier fights rebuild each one. One worker call per source fight plays all its samples; its rows go to
out/part-<source_episode_id>.parquet.
"""
import logging
import random
import sys
import time
from pathlib import Path

from apps.common.app import (TEACHER_KEYS, check_keys, exactly_when, main, required, run_json, snapshot,
                             teacher_settings, value_run, write_json)
from apps.common.replay import check_start, decision_rows, diverged, replay_requests
from runs import query
from apps.common.worker import run_parallel, run_worker, write_part
from environments.combat.schema import NAME

log = logging.getLogger(__name__)
BUILT = Path("build/main/fight_resample_worker")  # built by apps/common/job.sh
MAX_SAMPLES = 1000  # episode_id = source_episode_id * 1000 + k
RUN_KEYS = {"id", "query", "samples", "first_sample", "hp_sd", "random_potions", "value_run", "fights", "workers",
            "skip_diverged", *TEACHER_KEYS}
NET_LEAVES = ("value_net", "hybrid", "policy_net")  # need [run] value_run
COLUMNS = ["run_seed", "fight_index", "episode_id", "decision_index", "chosen_action", "ascension", "encounter", "floor",
           "starting_hp", "starting_max_hp"]


def inputs(config):
    """The source runs of the queried fights (each a bootstrap run: no inputs of its own), then the value run if any."""
    run = config["run"]
    sources = query.run_ids(run["query"])
    if not sources:
        sys.exit(f"no fights: {run['query']}")
    for source in sources:
        if run_json(source)["inputs"]:
            sys.exit(f"{source}: not a bootstrap run (it has inputs); only bootstrap fights replay")
    return sources + ([run["value_run"]] if "value_run" in run else [])


def samples(start, count, hp_sd, first):
    """Sample k (first <= k < first + count): episode_id = source * 1000 + k, starting HP ~ Normal(stored HP, hp_sd)
    rounded into [1, max HP]. A later run with a higher `first` plays new versions of the same fights."""
    result = []
    for k in range(first, first + count):
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
    # Rows record HP at the first decision: combat-start healing (e.g. Blood Vial) may raise it above the sampled HP.
    if [r["episode_id"] for r in firsts] != [s["episode_id"] for s in request["samples"]] or not all(
            s["starting_hp"] <= r["starting_hp"] <= r["starting_max_hp"] for r, s in zip(firsts, request["samples"])):
        raise RuntimeError(f"episode {episode}: replayed fights don't match the samples: "
                           f"{[(r['episode_id'], r['starting_hp']) for r in firsts]} vs {request['samples']}")
    for first in firsts:
        check_start(episode, first, start, ("encounter", "floor", "starting_max_hp"))
    write_part(out, episode, result["rows"])
    return result["teacher"], [{"starting_hp": r["starting_hp"], "won": r["won"], "final_hp": r["final_hp"],
                                "terminal_value": r["terminal_value"], "seconds": seconds / len(firsts)}
                               for r in firsts]


def resample(config, config_path, out):
    run = config["run"]
    check_keys(run, RUN_KEYS, "run")
    teacher = teacher_settings(run)
    exactly_when(run, "value_run", teacher["leaf"] in NET_LEAVES, f"with leaf {' / '.join(NET_LEAVES)}", "run")
    first = required(run, "first_sample", "run", int)
    count = required(run, "samples", "run", int)
    hp_sd = required(run, "hp_sd", "run", float)
    random_potions = required(run, "random_potions", "run", bool)
    workers = required(run, "workers", "run", int)
    if first < 0 or not 1 <= count <= MAX_SAMPLES - first:
        sys.exit(f"need first_sample >= 0 and 1 <= samples <= {MAX_SAMPLES} - first_sample")
    sources = [r for r in inputs(config) if r != run.get("value_run")]
    weights = None
    if "value_run" in run:
        weights, _ = snapshot(value_run(run["value_run"]).weights, out, "value_weights.bin")
    binary, worker_sha256 = snapshot(BUILT, out)
    episodes = sorted({r["episode_id"] for r in query.rows(run["query"], ["episode_id"], oracle=False)})
    if "fights" in run:  # only the first N queried fights (by episode_id)
        episodes = episodes[:required(run, "fights", "run", int)]
    fights = replay_requests(decision_rows(sources, COLUMNS, {e // 100 for e in episodes}), episodes)
    for request, start in fights.values():
        request.update(random_potions=random_potions, samples=samples(start, count, hp_sd, first), teacher=teacher)
    log.info("%d source fights x %d samples, hp_sd %s, random_potions %s, teacher %s, weights %s, %d workers -> %s",
             len(fights), count, hp_sd, random_potions, teacher, weights, workers, out)
    started, done, teacher, finished = time.monotonic(), [], None, 0

    def on_result(item, result):
        nonlocal teacher, finished
        teacher, results = result
        done.extend(results)
        finished += 1
        log.info("[%d/%d] episode %d: %s | total wins %d/%d", finished, len(fights), item[0],
                 " ".join(f"hp {r['starting_hp']}->{r['final_hp'] if r['won'] else 'lost'}" for r in results),
                 sum(r["won"] for r in done), len(done))

    skip_diverged = run.get("skip_diverged", False)
    if not isinstance(skip_diverged, bool):
        sys.exit("[run].skip_diverged must be true or false")
    skipped = []

    def play_or_skip(item):
        try:
            return play(item[0], *item[1], binary, weights, out)
        except RuntimeError as error:
            if skip_diverged and diverged(error):
                log.warning("episode %d skipped: replay diverged from the stored fight: %s", item[0], str(error)[-300:])
                skipped.append(item[0])
                return None, []
            raise

    def on_result_or_skip(item, result):
        if result[0] is not None:
            on_result(item, result)

    run_parallel(play_or_skip, fights.items(), workers, on_result_or_skip)
    if not done:
        sys.exit(f"no fight played ({len(skipped)} replays diverged)")
    summary = {"schema": NAME, "query": run["query"], "sources": sources, "value_run": run.get("value_run"),  # None: no net leaf
               "teacher": teacher, "worker_sha256": worker_sha256, "samples": count,
               "hp_sd": hp_sd, "random_potions": random_potions, "source_fights": len(fights), "fights": len(done),
               "wins": sum(r["won"] for r in done),
               "mean_terminal_value": sum(r["terminal_value"] for r in done) / len(done),
               "seconds_per_fight": sum(r["seconds"] for r in done) / len(done),
               "skipped_diverged": len(skipped), "skipped_episodes": skipped}
    write_json(out / "summary.json", summary)
    log.info("done in %.1f min: %s", (time.monotonic() - started) / 60, summary)


if __name__ == "__main__":
    main(resample, inputs)
