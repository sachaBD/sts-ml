#!/usr/bin/env python3
"""Play one seeded act 1 per seed in C++ (every combat teacher-searched) and write its rows as combat_v3 parquet."""
import itertools
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread, current_thread

from apps.common.app import FAIR_PLAY, ORACLE_BANNER, check_keys, flag, main, snapshot, write_json
from apps.common.worker import run_parallel, run_worker, write_part
from sts_combat_rl.schemas.combat_v3 import NAME

log = logging.getLogger(__name__)
BUILT = Path("build/main/bootstrap_fight_worker")  # built by apps/common/job.sh; each run plays with its own copy in out/
RUN_KEYS = {"id", "binary", "workers", "ascension", "oracle", "forever", "seeds", "first_seed", "status_seconds"}
# Run seeds stay below 2^40 so every derived episode_id fits int64: run_seed * 100 + fight_index here,
# and source_episode_id * 1000 + k in apps/fight_resample (< 2^40 * 10^5 ~ 1.1e17).
SEED_LIMIT = 2**40
STATUSES = ("other_boss", "died", "act_complete")  # what the worker reports per seed; other_boss = not played


@dataclass
class Run:
    seed: int
    status: str
    floor: int
    fights: int
    boss: bool  # reached the Slime Boss fight
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
            self.statuses[run.status] = self.statuses.get(run.status, 0) + 1
            self.fights += run.fights
            self.bosses += run.boss
            self.rows += run.rows
            self.teacher = run.teacher
            self.last_finished[current_thread().name] = time.monotonic()
        log.debug("%s", run)

    def summary(self, ascension, first_seed, oracle, worker_sha256):
        with self.lock:
            return {"schema": NAME, "oracle": oracle, "ascension": ascension, "first_seed": first_seed,
                    "worker_sha256": worker_sha256, "teacher": self.teacher, "runs": self.runs, **self.statuses,
                    "slime_fights": self.bosses, "fights": self.fights, "rows": self.rows}

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


def play(seed, binary, ascension, oracle, out, status):
    """One act 1 -> out/part-<seed>.parquet (no file for a run with no fights, e.g. other_boss).

    Rows arrive in play order: by fight_index, then decision rows, then child rows.
    """
    status.worker_started()
    if seed >= SEED_LIMIT:
        raise ValueError(f"run seed {seed} >= 2^40 (SEED_LIMIT)")
    start = time.monotonic()
    result = run_worker(binary, {"seed": seed, "ascension": ascension, "oracle": oracle})
    rows = result["rows"]
    if rows:
        write_part(out, seed, rows)
    status.record(Run(result["seed"], result["status"], result["floor"], result["fights"],
                      any(row["category"] == "boss" for row in rows), len(rows), time.monotonic() - start,
                      result["teacher"]))


def generate(config, config_path, out):
    run = config["run"]
    check_keys(run, RUN_KEYS, "run")
    # Random start so separate runs play different seeds; the lower half of [0, SEED_LIMIT) leaves room to count up.
    first_seed = run.get("first_seed", random.randrange(SEED_LIMIT // 2))
    seeds = itertools.count(first_seed) if run.get("forever") else range(first_seed, first_seed + run["seeds"])
    workers = run["workers"]
    ascension = run.get("ascension", 1)
    oracle = flag(run, "oracle")
    binary, worker_sha256 = snapshot(run.get("binary", BUILT), out, "bootstrap_fight_worker")
    start = time.monotonic()
    log.info("%s", ORACLE_BANNER if oracle else FAIR_PLAY)
    log.info("starting %s seeds from first_seed %d on %d workers -> %s",
             "unlimited" if run.get("forever") else run["seeds"], first_seed, workers, out)
    log.info("w0..w%d: seconds since that worker last finished a run", workers - 1)
    with Status(workers, run.get("status_seconds", 10), oracle) as status:
        # each worker takes the next seed as soon as its last run finishes (no batch barrier)
        run_parallel(lambda seed: play(seed, binary, ascension, oracle, out, status), seeds, workers)
    write_json(out / "summary.json", status.summary(ascension, first_seed, oracle, worker_sha256))
    log.info("%s", status.header())
    log.info("%s", status.line())
    log.info("done in %.1f min%s", (time.monotonic() - start) / 60, " [ORACLE run]" if oracle else "")


if __name__ == "__main__":
    main(generate)
