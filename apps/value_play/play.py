#!/usr/bin/env python3
"""Replay a value net's validation fights with a teacher leaf strategy; write them as combat_v3 parquet.

Spec: slop_docs/apps/value_play.md. The value run's combat_v3 inputs are the data runs; each replayed
fight must start in the same state as the stored one.
"""
import argparse
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import shutil

import msgpack
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sts_combat_rl.run import bootstrap_inputs, run_dir, run_parquet
from sts_combat_rl.schemas.combat_v3 import COMBAT_V3, NAME as COMBAT_V3_NAME
try:  # Direct script execution puts this directory on sys.path.
    from progress import Progress
except ModuleNotFoundError:  # Tests load this file as a module from the repository root.
    from apps.value_play.progress import Progress

log = logging.getLogger(__name__)
BUILT = Path("build-valexp/value_play_worker")  # built by job.sh; each run plays with its own copy in out/
NET_LEAVES = ("value_net", "hybrid")
# Same fight: the replayed decision_index 0 row must match the stored one on these columns.
START = ("encounter", "floor", "starting_hp", "starting_max_hp", "global_numeric", "cards", "monsters")
ORACLE_BANNER = ("\n" + "!" * 78 + "\n!!  ORACLE MODE: the teacher searches the TRUE state (perfect RNG / draw-order\n"
                 "!!  foresight). Upper-bound play, not fair play. Rows are tagged oracle = true.\n" + "!" * 78)
OUTCOME = ("won", "final_hp", "terminal_value")  # of the stored teacher's fight, logged next to the replay


def runs(config):
    """(value run, data runs) run_ids; the data runs are the value run's bootstrap inputs."""
    value_run = config["run"]["input"]
    return value_run, bootstrap_inputs(value_run)


def load_fights(data_runs, episodes):
    """episode_id -> (worker input, stored decision_index 0 row), across data runs."""
    seeds = sorted({e // 100 for e in episodes})
    rows = []
    for data_run in data_runs:
        table = ds.dataset(run_parquet(data_run), format="parquet").to_table(
            columns=["run_seed", "fight_index", "episode_id", "decision_index", "chosen_action", "ascension", *START, *OUTCOME],
            filter=(ds.field("row_kind") == "decision") & ds.field("run_seed").isin(seeds))
        rows.extend((data_run, row) for row in table.to_pylist())
    actions, starts = defaultdict(list), {}  # (run_seed, fight_index) -> chosen actions in play order
    for data_run, row in sorted(rows, key=lambda item: (item[1]["run_seed"], item[1]["fight_index"], item[1]["decision_index"])):
        actions[row["run_seed"], row["fight_index"]].append(row["chosen_action"])
        if row["decision_index"] == 0 and row["episode_id"] in episodes:
            if row["episode_id"] in starts:
                raise RuntimeError(f"episode {row['episode_id']}: requested from both {starts[row['episode_id']][0]} and {data_run}")
            starts[row["episode_id"]] = (data_run, row)
    missing = sorted(set(episodes) - set(starts))
    if missing:
        raise RuntimeError(f"{len(missing)} requested episodes missing from the data runs, e.g. {missing[:5]}")
    fights = {}
    for episode in episodes:
        seed, index = divmod(episode, 100)
        _, start = starts[episode]
        fights[episode] = ({"run_seed": seed, "ascension": start["ascension"], "fight_index": index,
                            "actions": [actions[seed, i] for i in range(index)]}, start)
    return fights


RUN_KEYS = {"id", "input", "workers", "leaf", "rollout_turns", "rollout_steps", "random_move", "episodes", "oracle"}


def teacher_config(run):
    """The worker's FIGHT.json teacher settings from [run]; the worker validates the combination."""
    if unknown := sorted(set(run) - RUN_KEYS):
        sys.exit(f"unknown [run] keys: {unknown}")
    teacher = {"leaf": run.get("leaf", "value_net"), "random_move": run.get("random_move", True)}
    teacher.update({k: run[k] for k in ("rollout_turns", "rollout_steps") if k in run})
    if run.get("oracle", False) is not False:
        if run["oracle"] is not True:
            sys.exit(f"oracle must be true or false, got {run['oracle']!r}")
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


def snapshot_worker(out):
    """Copy the built worker into the run dir (read-only) so a rebuild can't change this run; its sha256."""
    binary = out / "value_play_worker"
    shutil.copy2(BUILT, binary)
    binary.chmod(0o555)
    return binary.resolve(), hashlib.sha256(binary.read_bytes()).hexdigest()


def worker_argv(binary, weights, fight_json, tmp):
    """Worker command line; no weights (guided_rollout) -> --no-weights."""
    return [str(binary), str(weights) if weights else "--no-weights", str(fight_json), str(tmp)]


def play(episode, fight, start, binary, weights, out):
    """One fight -> out/part-<episode_id>.parquet; returns the worker's teacher settings and fight stats."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "fight.json").write_text(json.dumps(fight))
        began = time.monotonic()
        subprocess.run(worker_argv(binary, weights, tmp / "fight.json", tmp), check=True)
        seconds = time.monotonic() - began
        with (tmp / "fight.msgpack").open("rb") as source:
            result = msgpack.unpack(source, raw=False)
    table = pa.Table.from_pylist(result["rows"], schema=COMBAT_V3)
    first = table.slice(0, 1).to_pylist()[0]  # rows come in play order: decision_index 0 first
    differs = [c for c in START if first[c] != start[c]]
    if first["row_kind"] != "decision" or first["decision_index"] != 0 or differs:
        raise RuntimeError(f"episode {episode}: start state differs from the data run on {differs}")
    pq.write_table(table, out / f"part-{episode:08d}.parquet", compression="zstd")
    decisions = table.filter(pc.equal(table["row_kind"], "decision"))
    return result["teacher"], {
        "episode": episode, "encounter": first["encounter"], "floor": first["floor"],
        "starting_hp": first["starting_hp"], **{c: first[c] for c in OUTCOME},
        "turns": pc.max(decisions["turn"]).as_py(), "decisions": decisions.num_rows,
        "simulations": pc.sum(decisions["simulations_used"]).as_py(), "rows": table.num_rows, "seconds": seconds,
        "teacher": {c: start[c] for c in OUTCOME}}


def describe(fight, leaf):
    """One log line: the replay, then the stored teacher's result for the same fight."""
    result = lambda f: f"{'won ' if f['won'] else 'lost'} hp {f['final_hp']:>3} tv {f['terminal_value']:.3f}"
    return (f"episode {fight['episode']:>6} {fight['encounter']:<14} floor {fight['floor']:>2} "
            f"hp {fight['starting_hp']:>3}->  {leaf}: {result(fight)} | teacher: {result(fight['teacher'])} | "
            f"{fight['turns']:>2} turns {fight['decisions']:>3} decisions "
            f"{fight['simulations'] / fight['decisions']:>6,.0f} sims/decision {fight['seconds']:>4.0f}s")


def main(config_path, out):
    config = tomllib.loads(config_path.read_text())
    value_run, data_runs = runs(config)
    value = json.loads((run_dir(value_run) / "run.json").read_text())["summary"]
    value_out = run_dir(value_run) / "out"
    teacher_in = teacher_config(config["run"])
    leaf = teacher_in["leaf"]
    oracle = teacher_in.get("oracle", False)
    weights = value_out / value["weights"] if leaf in NET_LEAVES else None
    validation = json.loads((value_out / value["checkpoint_json"]).read_text())["validation_episode_ids"]
    episodes = select_episodes(config["run"], validation)
    fights = load_fights(data_runs, episodes)
    for fight, _ in fights.values():
        fight["teacher"] = teacher_in
    binary, sha256 = snapshot_worker(out)
    workers = config["run"]["workers"]
    log.info("%s", ORACLE_BANNER if oracle else "oracle: off (teacher searches sampled beliefs, fair play)")
    for line in ("", "Value play" + (" [ORACLE]" if oracle else ""), "==========", f"config:    {config_path}", f"output:    {out}",
                 f"value run: {value_run}", f"weights:   {weights}", f"data runs: {', '.join(data_runs)}",
                 f"teacher:   {json.dumps(teacher_in)}",
                 f"fights:    {len(fights)} of the value run's {len(validation)} validation episodes",
                 f"workers:   {workers}", f"worker:    {binary} (sha256 {sha256})", "",
                 f"Each line: the {leaf} teacher's replay | the stored teacher's result for the same fight", ""):
        log.info("%s", line)
    started, done, teacher = time.monotonic(), [], None
    progress = Progress(len(fights), leaf)
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(play, e, fight, start, binary, weights, out) for e, (fight, start) in fights.items()]
        try:
            for future in as_completed(futures):
                teacher, fight = future.result()
                done.append(fight)
                if progress.add(fight):
                    log.info("[%d/%d] wins %d vs teacher %d; mean terminal value %.3f vs %.3f",
                             len(done), len(fights), sum(f["won"] for f in done),
                             sum(f["teacher"]["won"] for f in done),
                             sum(f["terminal_value"] for f in done) / len(done),
                             sum(f["teacher"]["terminal_value"] for f in done) / len(done))
        except BaseException:
            log.exception("stopping: a fight failed; waiting for running workers")
            pool.shutdown(cancel_futures=True)
            raise
    progress.finish()
    mean = lambda values: sum(values) / len(values)
    summary = {"schema": COMBAT_V3_NAME, "value_run": value_run, "data_runs": data_runs, "teacher": teacher,
               "episodes": episodes, "worker": {"path": str(binary), "sha256": sha256},
               "fights": len(done), "wins": sum(f["won"] for f in done),
               "teacher_wins": sum(f["teacher"]["won"] for f in done), "rows": sum(f["rows"] for f in done),
               "mean_terminal_value": mean([f["terminal_value"] for f in done]),
               "teacher_mean_terminal_value": mean([f["teacher"]["terminal_value"] for f in done]),
               "seconds_per_fight": mean([f["seconds"] for f in done])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for line in ("", "Summary", "-------", f"teacher:        {json.dumps(teacher)}",
                 f"{'':16}{leaf:>14}{'teacher':>10}",
                 f"{'won':16}{summary['wins']:>14}{summary['teacher_wins']:>10}   of {summary['fights']}",
                 f"{'terminal value':16}{summary['mean_terminal_value']:>14.3f}{summary['teacher_mean_terminal_value']:>10.3f}   mean",
                 f"rows:           {summary['rows']:,}", f"seconds/fight:  {summary['seconds_per_fight']:.1f}",
                 f"wall time:      {(time.monotonic() - started) / 60:.1f} min", f"summary:        {out / 'summary.json'}"):
        log.info("%s", line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, help="run output dir")
    parser.add_argument("--inputs", action="store_true", help="print the value run and data run ids (for run.sh)")
    args = parser.parse_args()
    if args.inputs:
        value_run, data_runs = runs(tomllib.loads(args.config.read_text()))
        print(value_run, *data_runs, sep="\n")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
        main(args.config, args.out)
