#!/usr/bin/env python3
"""Play one seeded act 1 per seed in C++ (every combat teacher-searched) and write its rows as combat_v3 parquet.

Optional [run] keys choose the combat agent (default: the bootstrap teacher):
  leaf = "guided_rollout" (default) | "value_net" | "hybrid"; net leaves need value_run (a value_net_v1 run id),
  hybrid needs rollout_turns / rollout_steps; random_move = true (default) plays one random move per fight.
summary.json lists every seed's outcome under "results" (query.py view act1_results).
"""
import itertools
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread, current_thread

from apps.common.app import (FAIR_PLAY, ORACLE_BANNER, check_keys, exactly_when, main, required, snapshot,
                             value_run, write_json)
from apps.common.worker import run_parallel, run_worker, write_part
from sts_combat_rl.schemas.combat_v3 import NAME

log = logging.getLogger(__name__)
BUILT = Path("build/main/bootstrap_fight_worker")  # built by apps/common/job.sh; each run plays with its own copy in out/
RUN_KEYS = {"id", "workers", "ascension", "oracle", "forever", "seeds", "first_seed", "bosses",
            "stop_factor", "merge_identical_cards", "simulations", "leaf", "value_run", "rollout_turns",
            "rollout_steps", "random_move"}
LEAVES = ("guided_rollout", "value_net", "hybrid")
CATEGORIES = {"easy", "hard", "elite", "event", "boss"}
STATUS_SECONDS = 10  # progress table interval
ACT1_BOSSES = ("slime_boss", "the_guardian", "hexaghost")


def selected_bosses(run):
    """Nonempty subset of act 1 bosses."""
    bosses = required(run, "bosses", "run", list)
    if (not isinstance(bosses, list) or not bosses or any(type(boss) is not str or boss not in ACT1_BOSSES for boss in bosses)
            or len(bosses) != len(set(bosses))):
        sys.exit(f"bosses must be a nonempty list of unique names from {ACT1_BOSSES}")
    return bosses


def teacher_options(run):
    """Opt-in search tweaks; the worker also validates the request."""
    if "stop_factor" in run and (type(run["stop_factor"]) not in (int, float) or
                                 not 0 < run["stop_factor"] <= 1):
        sys.exit("stop_factor must be a number in (0, 1]")
    if "merge_identical_cards" in run and type(run["merge_identical_cards"]) is not bool:
        sys.exit("merge_identical_cards must be true or false")
    return {key: run[key] for key in ("stop_factor", "merge_identical_cards") if key in run}


def simulation_budgets(run):
    """Search simulations per decision, for every fight category."""
    budgets = required(run, "simulations", "run", dict)
    check_keys(budgets, CATEGORIES, "run.simulations")
    for category in sorted(CATEGORIES):
        if required(budgets, category, "run.simulations", int) < 1:
            sys.exit(f"simulations.{category} must be a positive integer")
    return budgets


def leaf_settings(run):
    """The worker's leaf {kind, rollout_turns, rollout_steps} and the value run it needs (None for guided_rollout)."""
    kind = run.get("leaf", "guided_rollout")
    if kind not in LEAVES:
        sys.exit(f"leaf must be one of {LEAVES}")
    exactly_when(run, "value_run", kind != "guided_rollout", "with a net leaf (value_net, hybrid)", "run")
    leaf = {"kind": kind}
    for key in ("rollout_turns", "rollout_steps"):
        exactly_when(run, key, kind == "hybrid", "with leaf hybrid", "run")
        if kind == "hybrid" and required(run, key, "run", int) < 1:
            sys.exit(f"{key} must be a positive integer")
        if kind == "hybrid":
            leaf[key] = run[key]
    return leaf, run.get("value_run")


def random_move(run):
    """One uniformly random move per fight (training-data diversity); default true, as the bootstrap teacher."""
    return required(run, "random_move", "run", bool) if "random_move" in run else True


def inputs(config):
    """The value run whose net plays, if any (run lineage)."""
    return [config["run"]["value_run"]] if "value_run" in config["run"] else []


# Run seeds stay below 2^40 so every derived episode_id fits int64: run_seed * 100 + fight_index here,
# and source_episode_id * 1000 + k in apps/fight_resample (< 2^40 * 10^5 ~ 1.1e17).
SEED_LIMIT = 2**40
STATUSES = ("other_boss", "died", "act_complete")  # what the worker reports per seed; other_boss = not played


@dataclass
class Run:
    seed: int
    act_boss: str  # the seed's act 1 boss
    status: str
    floor: int
    fights: int
    final_hp: int
    boss: bool  # reached the act 1 boss fight
    rows: int
    seconds: float
    teacher: dict  # search settings (agents/teacher_search.cpp)

    def __str__(self):
        return f"seed {self.seed}: {self.status} on floor {self.floor}, {self.fights} fights, {self.rows} rows, {self.seconds:.1f}s"


class Status:
    """Run totals and each worker's time since its last finished run, logged as a table every `interval` seconds."""

    HEADER_EVERY = 20

    def __init__(self, workers, interval, oracle):
        self.workers = workers
        self.oracle = oracle
        self.runs = self.fights = self.bosses = self.rows = 0
        self.statuses = dict.fromkeys(STATUSES, 0)
        self.last_finished = {}  # worker thread name -> monotonic time
        self.teacher = None
        self.results = []  # one per seed (summary.json "results")
        self.lock = Lock()
        self.stopped = Event()
        self.reporter = Thread(target=self.report_every, args=(interval,), daemon=True)

    def __enter__(self):
        self.reporter.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.reporter.join()

    def worker_started(self):
        with self.lock:
            self.last_finished.setdefault(current_thread().name, time.monotonic())

    def record(self, run):
        with self.lock:
            self.runs += run.status != "other_boss"  # a run = a seed actually played
            self.statuses[run.status] += 1
            self.fights += run.fights
            self.bosses += run.boss
            self.rows += run.rows
            self.teacher = run.teacher
            self.results.append({"seed": run.seed, "boss": run.act_boss, "status": run.status, "floor": run.floor,
                                 "fights": run.fights, "reached_boss": run.boss,
                                 "won_boss": run.status == "act_complete", "final_hp": run.final_hp})
            self.last_finished[current_thread().name] = time.monotonic()
        log.debug("%s", run)

    def summary(self, ascension, first_seed, oracle, worker_sha256):
        with self.lock:
            return {"schema": NAME, "oracle": oracle, "ascension": ascension, "first_seed": first_seed,
                    "worker_sha256": worker_sha256, "teacher": self.teacher, "runs": self.runs, **self.statuses,
                    "boss_fights": self.bosses, "fights": self.fights, "rows": self.rows,
                    "results": sorted(self.results, key=lambda r: r["seed"])}

    def header(self):
        workers = "".join(f"{f'w{i}':>5}" for i in range(self.workers))
        tag = "[ORACLE] " if self.oracle else ""
        return f"{tag}{'runs':>7} {'skipped':>8} {'died':>6} {'boss':>6} {'cleared':>7} {'fights':>7} {'rows':>10}  {workers}"

    def line(self):
        now = time.monotonic()
        with self.lock:
            s = self.statuses
            since = [f"{now - last:>5.0f}" for last in self.last_finished.values()]
            since += [f"{'-':>5}"] * (self.workers - len(since))
            return (f"{self.runs:>7,} {s['other_boss']:>8,} {s['died']:>6,} {self.bosses:>6,} {s['act_complete']:>7,} "
                    f"{self.fights:>7,} {self.rows:>10,}  {''.join(since)}")

    def report_every(self, interval):
        for count in itertools.count():
            if self.stopped.wait(interval):
                return
            if count % self.HEADER_EVERY == 0:
                log.info("%s", self.header())
            log.info("%s", self.line())


def play(seed, binary, weights, request, out, status):
    """One act 1 -> out/part-<seed>.parquet (no file for a run with no fights, e.g. other_boss).

    Rows arrive in play order: by fight_index, then decision rows, then child rows.
    """
    status.worker_started()
    if seed >= SEED_LIMIT:
        raise ValueError(f"run seed {seed} >= 2^40 (SEED_LIMIT)")
    start = time.monotonic()
    result = run_worker(binary, {**request, "seed": seed}, weights)
    rows = result["rows"]
    if rows:
        write_part(out, seed, rows)
    status.record(Run(result["seed"], result["boss"], result["status"], result["floor"], result["fights"],
                      result["final_hp"],
                      any(row["category"] == "boss" for row in rows), len(rows), time.monotonic() - start,
                      result["teacher"]))


def generate(config, config_path, out):
    run = config["run"]
    check_keys(run, RUN_KEYS, "run")
    # Separate runs should play different seeds; keep first_seed in the lower half of [0, SEED_LIMIT) to count up.
    first_seed = required(run, "first_seed", "run", int)
    forever = "forever" in run
    if forever == ("seeds" in run) or (forever and run["forever"] is not True):
        sys.exit("set exactly one of [run] forever = true / seeds = N")
    seeds = itertools.count(first_seed) if forever else range(first_seed, first_seed + required(run, "seeds", "run", int))
    workers = required(run, "workers", "run", int)
    ascension = required(run, "ascension", "run", int)
    oracle = required(run, "oracle", "run", bool)
    bosses = selected_bosses(run)
    teacher = teacher_options(run)
    simulations = simulation_budgets(run)
    leaf, value_id = leaf_settings(run)
    moves = random_move(run)
    weights = snapshot(value_run(value_id).weights, out, "value_weights.bin")[0] if value_id else None
    request = {"ascension": ascension, "oracle": oracle, "bosses": bosses, "teacher": teacher,
               "simulations": simulations, "leaf": leaf, "random_move": moves}
    binary, worker_sha256 = snapshot(BUILT, out)
    start = time.monotonic()
    log.info("%s", ORACLE_BANNER if oracle else FAIR_PLAY)
    log.info("starting %s seeds from first_seed %d on %d workers; bosses: %s -> %s",
             "unlimited" if forever else run["seeds"], first_seed, workers, ", ".join(bosses), out)
    log.info("teacher tweaks: %s; simulations by category: %s", teacher or "none", simulations)
    log.info("leaf: %s%s; random move: %s", leaf, f" (value run {value_id}, weights {weights})" if value_id else "", moves)
    log.info("w0..w%d: seconds since that worker last finished a run", workers - 1)
    with Status(workers, STATUS_SECONDS, oracle) as status:
        # each worker takes the next seed as soon as its last run finishes (no batch barrier)
        run_parallel(lambda seed: play(seed, binary, weights, request, out, status),
                     seeds, workers)
    write_json(out / "summary.json", {**status.summary(ascension, first_seed, oracle, worker_sha256), "bosses": bosses,
                                          "value_run": value_id})
    log.info("%s", status.header())
    log.info("%s", status.line())
    log.info("done in %.1f min%s", (time.monotonic() - start) / 60, " [ORACLE run]" if oracle else "")


if __name__ == "__main__":
    main(generate, inputs)
