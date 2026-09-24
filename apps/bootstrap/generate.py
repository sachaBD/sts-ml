#!/usr/bin/env python3
"""Play one seeded act 1 per seed in C++ (every combat teacher-searched) and write its rows as combat_v3 parquet."""
import argparse
import itertools
import json
import logging
import random
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Event, Lock, Thread, current_thread

import msgpack
import pyarrow as pa
import pyarrow.parquet as pq
from sts_combat_rl.schemas.combat_v3 import COMBAT_V3, NAME

log = logging.getLogger(__name__)
ORACLE_BANNER = ("\n" + "!" * 78 + "\n!!  ORACLE MODE: the teacher searches the TRUE state (perfect RNG / draw-order\n"
                 "!!  foresight). Upper-bound data, not fair play. Rows are tagged oracle = true.\n" + "!" * 78)
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
            self.last_finished[current_thread().name] = time.monotonic()

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

    def summary(self, ascension, first_seed, oracle):
        with self.lock:
            return {"schema": NAME, "oracle": oracle, "ascension": ascension, "first_seed": first_seed, "teacher": self.teacher, "runs": self.runs, **self.statuses, "slime_fights": self.bosses, "fights": self.fights, "rows": self.rows}

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


def to_parquet(msgpack_path, part, seconds):
    """Rows arrive in play order: by fight_index, then decision rows, then child rows."""
    with msgpack_path.open("rb") as source:
        result = msgpack.unpack(source, raw=False)
    rows = result["rows"]
    if rows:
        pq.write_table(pa.Table.from_pylist(rows, schema=COMBAT_V3), part, compression="zstd")
    return Run(result["seed"], result["status"], result["floor"], result["fights"],
               any(row["category"] == "boss" for row in rows), len(rows), seconds, result["teacher"])


def play(seed, binary, ascension, oracle, out, status):
    """One act 1 -> out/part-<seed>.parquet (no file for a run with no fights, e.g. other_boss)."""
    with tempfile.TemporaryDirectory() as tmp:
        start = time.monotonic()
        subprocess.run([binary, str(seed), str(ascension), tmp, str(int(oracle))], check=True)
        status.record(to_parquet(Path(tmp) / "fight.msgpack", out / f"part-{seed:06d}.parquet", time.monotonic() - start))


def generate(config, out):
    run = tomllib.loads(config.read_text())["run"]
    # Random start so separate runs play different seeds; the lower half of [0, SEED_LIMIT) leaves room to count up.
    first_seed = run.get("first_seed", random.randrange(SEED_LIMIT // 2))
    seeds = itertools.count(first_seed) if run.get("forever") else range(first_seed, first_seed + run["seeds"])
    workers = run["workers"]
    ascension = run.get("ascension", 1)
    oracle = run.get("oracle", False)
    if not isinstance(oracle, bool):
        raise ValueError(f"oracle must be true or false, got {oracle!r}")
    start = time.monotonic()
    log.info("%s", ORACLE_BANNER if oracle else "oracle: off (teacher searches sampled beliefs, fair play)")
    log.info("starting %s seeds from first_seed %d on %d workers -> %s",
             "unlimited" if run.get("forever") else run["seeds"], first_seed, workers, out)
    log.info("w0..w%d: seconds since that worker last finished a run", workers - 1)
    with Status(workers, run.get("status_seconds", 10), oracle) as status, ThreadPoolExecutor(
            workers, thread_name_prefix="w", initializer=status.worker_started) as pool:
        run_seed = partial(play, binary=run["binary"], ascension=ascension, oracle=oracle, out=out, status=status)
        seeds, seeds_lock = iter(seeds), Lock()

        def work(_):  # each worker takes the next seed as soon as its last run finishes (no batch barrier)
            while True:
                with seeds_lock:
                    seed = next(seeds, None)
                if seed is None:
                    return
                if seed >= SEED_LIMIT:
                    raise ValueError(f"run seed {seed} >= 2^40 (SEED_LIMIT)")
                run_seed(seed)

        list(pool.map(work, range(workers)))
    (out / "summary.json").write_text(json.dumps(status.summary(ascension, first_seed, oracle), indent=2) + "\n")
    log.info("%s", status.header())
    log.info("%s", status.line())
    log.info("done in %.1f min%s", (time.monotonic() - start) / 60, " [ORACLE run]" if oracle else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    generate(args.config, args.out)
