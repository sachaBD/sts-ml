#!/usr/bin/env python3
"""Replay a value net's validation fights with a teacher leaf strategy; write them as combat_v3 parquet.

The value run's combat_v3 inputs are the data runs; each replayed
fight must start in the same state as the stored one.
"""
import json
import logging
import sys
import time
from pathlib import Path

import pyarrow.compute as pc
from apps.common.app import (FAIR_PLAY, ORACLE_BANNER, TEACHER_KEYS, check_keys, main, required, run_json, snapshot,
                             teacher_settings, value_run, write_json)
from apps.common.replay import COLUMNS, check_start, decision_rows, diverged, replay_requests
from apps.common.worker import run_parallel, run_worker, write_part
from apps.value_play.progress import Progress
from sts_combat_rl import query
from sts_combat_rl.run import bootstrap_inputs
from sts_combat_rl.schemas.combat_v3 import NAME as COMBAT_V3_NAME

log = logging.getLogger(__name__)
BUILT = Path("build/main/value_play_worker")  # built by apps/common/job.sh; each run plays with its own copy in out/
NET_LEAVES = ("value_net", "hybrid", "policy_net")
OUTCOME = ("won", "final_hp", "terminal_value")  # of the stored teacher's fight, logged next to the replay
RUN_KEYS = {"id", "input", "workers", "episodes", "query", "skip_diverged", *TEACHER_KEYS}


def inputs(config):
    """The value run, then the data runs: those of [run] query if given, else the value run's bootstrap inputs."""
    run = config["run"]
    value = run["input"]
    if "query" not in run:
        return [value, *bootstrap_inputs(value)]
    sources = query.run_ids(run["query"])
    if not sources:
        sys.exit(f"no fights: {run['query']}")
    if not_bootstrap := [s for s in sources if run_json(s)["inputs"]]:
        sys.exit(f"query fights must come from bootstrap runs (no inputs of their own): {not_bootstrap}")
    return [value, *sources]


def query_episodes(run, meta):
    """[run] query: its original fights (source_episode_id null); none may be a checkpoint training fight or deck."""
    rows = query.rows(run["query"], ["episode_id", "run_seed"], "source_episode_id is null", oracle=False)
    episodes = sorted({r["episode_id"] for r in rows})
    train_ids, train_seeds = set(meta["train_episode_ids"]), set(meta["train_run_seeds"])
    if leaked := sorted({r["episode_id"] for r in rows if r["episode_id"] in train_ids or r["run_seed"] in train_seeds}):
        sys.exit(f"{len(leaked)} query fights are checkpoint training fights/decks, e.g. {leaked[:5]}")
    if not episodes:
        sys.exit(f"no fights: {run['query']}")
    return episodes


def select_episodes(run, validation):
    """[run] episodes: "validation" (all the checkpoint's validation episodes) or a list of them."""
    episodes = validation if run["episodes"] == "validation" else run["episodes"]
    if not isinstance(episodes, list) or not all(isinstance(e, int) for e in episodes):
        sys.exit("episodes must be integer episode_ids")
    outside = sorted(set(episodes) - set(validation))
    if outside or len(set(episodes)) != len(episodes) or not episodes:
        sys.exit(f"episodes must be distinct checkpoint validation episodes; not validation: {outside}")
    return episodes


def play_or_skip(episode, request, start, binary, weights, out, skip_diverged):
    """play(), or with skip_diverged (None, {"episode", "skipped": reason}) for a fight whose replay diverged."""
    try:
        return play(episode, request, start, binary, weights, out)
    except RuntimeError as error:
        if skip_diverged and diverged(error):
            log.warning("episode %d skipped: replay diverged from the stored fight: %s", episode, str(error)[-300:])
            return None, {"episode": episode, "skipped": str(error)[-300:]}
        raise


def play(episode, request, start, binary, weights, out):
    """One fight -> out/part-<episode_id>.parquet; returns the worker's teacher settings and fight stats."""
    began = time.monotonic()
    result = run_worker(binary, request, weights)
    seconds = time.monotonic() - began
    first = result["rows"][0]  # rows come in play order: decision_index 0 first
    if first["row_kind"] != "decision" or first["decision_index"] != 0:
        raise RuntimeError(f"episode {episode}: the first row is not decision 0")
    check_start(episode, first, start)
    table = write_part(out, episode, result["rows"])
    decisions = table.filter(pc.equal(table["row_kind"], "decision"))
    return result["teacher"], {
        "episode": episode, "encounter": first["encounter"], "floor": first["floor"],
        "starting_hp": first["starting_hp"], **{c: first[c] for c in OUTCOME},
        "turns": pc.max(decisions["turn"]).as_py(), "decisions": decisions.num_rows,
        "simulations": pc.sum(decisions["simulations_used"]).as_py(), "rows": table.num_rows, "seconds": seconds,
        "search_seconds": result.get("search_seconds"),
        "teacher": {c: start[c] for c in OUTCOME}}


def replay(config, config_path, out):
    value_id, *data_runs = inputs(config)
    value = value_run(value_id)
    run = config["run"]
    check_keys(run, RUN_KEYS, "run")
    if ("episodes" in run) == ("query" in run):
        sys.exit("set exactly one of [run] episodes / query")
    teacher_in = teacher_settings(run)
    leaf = teacher_in["leaf"]
    oracle = teacher_in["oracle"]
    label = f"{leaf} ORACLE" if oracle else leaf  # log label only
    weights = snapshot(value.weights, out, "value_weights.bin")[0] if leaf in NET_LEAVES else None
    validation = value.meta["validation_episode_ids"]
    episodes = query_episodes(run, value.meta) if "query" in run else select_episodes(run, validation)
    rows = decision_rows(data_runs, (*COLUMNS, *OUTCOME), {e // 100 for e in episodes})
    fights = replay_requests(rows, episodes)
    for request, _ in fights.values():
        request["teacher"] = teacher_in
    binary, sha256 = snapshot(BUILT, out)
    workers = required(run, "workers", "run", int)
    skip_diverged = run.get("skip_diverged", False)
    if not isinstance(skip_diverged, bool):
        sys.exit("[run].skip_diverged must be true or false")
    log.info("%s", ORACLE_BANNER if oracle else FAIR_PLAY)
    for line in ("", "Value play" + (" [ORACLE]" if oracle else ""), "==========", f"config:    {config_path}", f"output:    {out}",
                 f"value run: {value_id}", f"weights:   {weights}", f"data runs: {', '.join(data_runs)}",
                 f"teacher:   {json.dumps(teacher_in)}",
                 (f"fights:    {len(fights)} from query {run['query']!r}" if "query" in run
                  else f"fights:    {len(fights)} of the value run's {len(validation)} validation episodes"),
                 f"workers:   {workers}", f"worker:    {binary} (sha256 {sha256})", "",
                 f"Each line: the {label} teacher's replay | the stored teacher's result for the same fight", ""):
        log.info("%s", line)
    started, done, skipped, teacher = time.monotonic(), [], [], None
    progress = Progress(len(fights), label)

    def on_result(_, result):
        nonlocal teacher
        settings, fight = result
        if "skipped" in fight:
            skipped.append(fight)
            return
        teacher = settings
        done.append(fight)
        if progress.add(fight):
            log.info("[%d/%d] wins %d vs teacher %d; mean terminal value %.3f vs %.3f",
                     len(done), len(fights), sum(f["won"] for f in done),
                     sum(f["teacher"]["won"] for f in done),
                     sum(f["terminal_value"] for f in done) / len(done),
                     sum(f["teacher"]["terminal_value"] for f in done) / len(done))

    run_parallel(lambda item: play_or_skip(item[0], *item[1], binary, weights, out, skip_diverged), fights.items(),
                 workers, on_result)
    progress.finish()
    if not done:
        sys.exit(f"every fight's replay diverged ({len(skipped)} skipped)")
    mean = lambda values: sum(values) / len(values)
    summary = {"schema": COMBAT_V3_NAME, "value_run": value_id, "data_runs": data_runs, "teacher": teacher,
               "episodes": episodes, "worker": {"path": str(binary), "sha256": sha256},
               "fights": len(done), "wins": sum(f["won"] for f in done),
               "teacher_wins": sum(f["teacher"]["won"] for f in done), "rows": sum(f["rows"] for f in done),
               "mean_terminal_value": mean([f["terminal_value"] for f in done]),
               "teacher_mean_terminal_value": mean([f["teacher"]["terminal_value"] for f in done]),
               "seconds_per_fight": mean([f["seconds"] for f in done]),
               "skipped_diverged": len(skipped), "skipped": skipped,
               "per_fight": [{k: f[k] for k in ("episode", "decisions", "simulations", "seconds", "search_seconds")}
                             for f in done]}
    write_json(out / "summary.json", summary)
    for line in ("", "Summary", "-------", f"teacher:        {json.dumps(teacher)}",
                 f"{'':16}{label:>22}{'teacher':>10}",
                 f"{'won':16}{summary['wins']:>22}{summary['teacher_wins']:>10}   of {summary['fights']}",
                 f"{'terminal value':16}{summary['mean_terminal_value']:>22.3f}{summary['teacher_mean_terminal_value']:>10.3f}   mean",
                 f"rows:           {summary['rows']:,}", f"seconds/fight:  {summary['seconds_per_fight']:.1f}",
                 f"skipped:        {len(skipped)} fights whose replay diverged from the stored fight",
                 f"wall time:      {(time.monotonic() - started) / 60:.1f} min", f"summary:        {out / 'summary.json'}"):
        log.info("%s", line)


if __name__ == "__main__":
    main(replay, inputs)
