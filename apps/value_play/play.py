#!/usr/bin/env python3
"""Replay a value net's validation fights with a teacher leaf strategy; write them as combat_v3 parquet.

Spec: slop_docs/apps/value_play.md. The value run's combat_v3 inputs are the data runs; each replayed
fight must start in the same state as the stored one.
"""
import json
import logging
import sys
import time
from pathlib import Path

import pyarrow.compute as pc
from apps.common.app import FAIR_PLAY, ORACLE_BANNER, check_keys, flag, main, snapshot, value_run, write_json
from apps.common.replay import COLUMNS, check_start, decision_rows, replay_requests
from apps.common.worker import run_parallel, run_worker, write_part
from apps.value_play.progress import Progress
from sts_combat_rl.run import bootstrap_inputs
from sts_combat_rl.schemas.combat_v3 import NAME as COMBAT_V3_NAME

log = logging.getLogger(__name__)
BUILT = Path("build/valexp/value_play_worker")  # built by apps/common/job.sh; each run plays with its own copy in out/
NET_LEAVES = ("value_net", "hybrid")
OUTCOME = ("won", "final_hp", "terminal_value")  # of the stored teacher's fight, logged next to the replay
RUN_KEYS = {"id", "input", "workers", "leaf", "rollout_turns", "rollout_steps", "random_move", "episodes", "oracle"}


def inputs(config):
    """The value run, then its bootstrap combat_v3 input runs (the data runs)."""
    value = config["run"]["input"]
    return [value, *bootstrap_inputs(value)]


def teacher_config(run):
    """The worker request's teacher settings from [run]; the worker validates the combination."""
    check_keys(run, RUN_KEYS, "run")
    teacher = {"leaf": run.get("leaf", "value_net"), "random_move": run.get("random_move", True)}
    teacher.update({k: run[k] for k in ("rollout_turns", "rollout_steps") if k in run})
    if flag(run, "oracle"):
        teacher["oracle"] = True
    return teacher


def select_episodes(run, validation):
    """[run] episodes (default: all validation episodes); each must be a validation episode."""
    episodes = run.get("episodes", validation)
    if not all(isinstance(e, int) for e in episodes):
        sys.exit("episodes must be integer episode_ids")
    outside = sorted(set(episodes) - set(validation))
    if outside or len(set(episodes)) != len(episodes) or not episodes:
        sys.exit(f"episodes must be distinct checkpoint validation episodes; not validation: {outside}")
    return episodes


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
        "teacher": {c: start[c] for c in OUTCOME}}


def replay(config, config_path, out):
    value_id, *data_runs = inputs(config)
    value = value_run(value_id)
    teacher_in = teacher_config(config["run"])
    leaf = teacher_in["leaf"]
    oracle = teacher_in.get("oracle", False)
    label = f"{leaf} ORACLE" if oracle else leaf  # log label only
    weights = snapshot(value.weights, out, "value_weights.bin")[0] if leaf in NET_LEAVES else None
    validation = value.meta["validation_episode_ids"]
    episodes = select_episodes(config["run"], validation)
    rows = decision_rows(data_runs, (*COLUMNS, *OUTCOME), {e // 100 for e in episodes})
    fights = replay_requests(rows, episodes)
    for request, _ in fights.values():
        request["teacher"] = teacher_in
    binary, sha256 = snapshot(BUILT, out)
    workers = config["run"]["workers"]
    log.info("%s", ORACLE_BANNER if oracle else FAIR_PLAY)
    for line in ("", "Value play" + (" [ORACLE]" if oracle else ""), "==========", f"config:    {config_path}", f"output:    {out}",
                 f"value run: {value_id}", f"weights:   {weights}", f"data runs: {', '.join(data_runs)}",
                 f"teacher:   {json.dumps(teacher_in)}",
                 f"fights:    {len(fights)} of the value run's {len(validation)} validation episodes",
                 f"workers:   {workers}", f"worker:    {binary} (sha256 {sha256})", "",
                 f"Each line: the {label} teacher's replay | the stored teacher's result for the same fight", ""):
        log.info("%s", line)
    started, done, teacher = time.monotonic(), [], None
    progress = Progress(len(fights), label)

    def on_result(_, result):
        nonlocal teacher
        teacher, fight = result
        done.append(fight)
        if progress.add(fight):
            log.info("[%d/%d] wins %d vs teacher %d; mean terminal value %.3f vs %.3f",
                     len(done), len(fights), sum(f["won"] for f in done),
                     sum(f["teacher"]["won"] for f in done),
                     sum(f["terminal_value"] for f in done) / len(done),
                     sum(f["teacher"]["terminal_value"] for f in done) / len(done))

    run_parallel(lambda item: play(item[0], *item[1], binary, weights, out), fights.items(), workers, on_result)
    progress.finish()
    mean = lambda values: sum(values) / len(values)
    summary = {"schema": COMBAT_V3_NAME, "value_run": value_id, "data_runs": data_runs, "teacher": teacher,
               "episodes": episodes, "worker": {"path": str(binary), "sha256": sha256},
               "fights": len(done), "wins": sum(f["won"] for f in done),
               "teacher_wins": sum(f["teacher"]["won"] for f in done), "rows": sum(f["rows"] for f in done),
               "mean_terminal_value": mean([f["terminal_value"] for f in done]),
               "teacher_mean_terminal_value": mean([f["teacher"]["terminal_value"] for f in done]),
               "seconds_per_fight": mean([f["seconds"] for f in done])}
    write_json(out / "summary.json", summary)
    for line in ("", "Summary", "-------", f"teacher:        {json.dumps(teacher)}",
                 f"{'':16}{label:>22}{'teacher':>10}",
                 f"{'won':16}{summary['wins']:>22}{summary['teacher_wins']:>10}   of {summary['fights']}",
                 f"{'terminal value':16}{summary['mean_terminal_value']:>22.3f}{summary['teacher_mean_terminal_value']:>10.3f}   mean",
                 f"rows:           {summary['rows']:,}", f"seconds/fight:  {summary['seconds_per_fight']:.1f}",
                 f"wall time:      {(time.monotonic() - started) / 60:.1f} min", f"summary:        {out / 'summary.json'}"):
        log.info("%s", line)


if __name__ == "__main__":
    main(replay, inputs)
