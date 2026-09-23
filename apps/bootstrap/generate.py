#!/usr/bin/env python3
"""Play one seeded act 1 per seed in C++ (every combat teacher-searched) and write its rows as combat_v3 parquet."""
import argparse
import itertools
import json
import logging
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import batched
from pathlib import Path
from threading import Event, Lock, Thread, current_thread

import msgpack
import pyarrow as pa
import pyarrow.parquet as pq
from schema import COMBAT_V3, NAME

log = logging.getLogger(__name__)
STATUSES = ("other_boss", "died", "act_complete")  # what the worker reports per seed; other_boss = not played


@dataclass
class Run:
    seed: int
    status: str
    floor: int
    fights: int
    rows: int
    seconds: float

    def __str__(self):
        return f"seed {self.seed}: {self.status} on floor {self.floor}, {self.fights} fights, {self.rows} rows, {self.seconds:.1f}s"


class Status:
    """Run totals and each worker's time since its last finished run, logged as a table every `interval` seconds."""

    HEADER_EVERY = 20

    def __init__(self, workers, interval):
        self.workers = workers
        self.runs = self.fights = self.rows = 0
        self.statuses = dict.fromkeys(STATUSES, 0)
        self.last_finished = {}  # worker thread name -> monotonic time
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
            self.rows += run.rows
            self.last_finished[current_thread().name] = time.monotonic()
        log.debug("%s", run)

    def summary(self):
        with self.lock:
            return {"schema": NAME, "runs": self.runs, **self.statuses, "fights": self.fights, "rows": self.rows}

    def header(self):
        workers = "".join(f"{f'w{i}':>5}" for i in range(self.workers))
        return f"{'runs':>7} {'skipped':>8} {'died':>6} {'cleared':>7} {'fights':>7} {'rows':>10}  {workers}"

    def line(self):
        now = time.monotonic()
        with self.lock:
            s = self.statuses
            since = [f"{now - last:>5.0f}" for last in self.last_finished.values()]
            since += [f"{'-':>5}"] * (self.workers - len(since))
            return (f"{self.runs:>7,} {s['other_boss']:>8,} {s['died']:>6,} {s['act_complete']:>7,} "
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
    return Run(result["seed"], result["status"], result["floor"], result["fights"], len(rows), seconds)


def play(seed, binary, ascension, out, status):
    """One act 1 -> out/part-<seed>.parquet (no file for a run with no fights, e.g. other_boss)."""
    with tempfile.TemporaryDirectory() as tmp:
        start = time.monotonic()
        subprocess.run([binary, str(seed), str(ascension), tmp], check=True)
        status.record(to_parquet(Path(tmp) / "fight.msgpack", out / f"part-{seed:06d}.parquet", time.monotonic() - start))


def generate(config, out):
    run = tomllib.loads(config.read_text())["run"]
    seeds = itertools.count() if run.get("forever") else range(run["seeds"])
    workers = run["workers"]
    start = time.monotonic()
    log.info("starting %s seeds on %d workers -> %s", "unlimited" if run.get("forever") else run["seeds"], workers, out)
    log.info("w0..w%d: seconds since that worker last finished a run", workers - 1)
    with Status(workers, run.get("status_seconds", 10)) as status, ThreadPoolExecutor(
            workers, thread_name_prefix="w", initializer=status.worker_started) as pool:
        run_seed = partial(play, binary=run["binary"], ascension=run.get("ascension", 1), out=out, status=status)
        for batch in batched(seeds, workers):
            for _ in pool.map(run_seed, batch):
                pass
    (out / "summary.json").write_text(json.dumps(status.summary(), indent=2) + "\n")
    log.info("%s", status.header())
    log.info("%s", status.line())
    log.info("done in %.1f min", (time.monotonic() - start) / 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    generate(args.config, args.out)
