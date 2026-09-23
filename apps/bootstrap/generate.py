#!/usr/bin/env python3
"""Run one C++ fight per seed and convert each result to parquet."""
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
from schema import COMBAT_V2

log = logging.getLogger(__name__)
FIGHT = {"act": 1, "floor": 16, "encounter": "slime_boss"}  # the only fight the worker plays


@dataclass
class Fight:
    seed: int
    status: str
    won: bool
    rows: int
    seconds: float

    def __str__(self):
        outcome = f"boss {'killed' if self.won else 'survived'}" if self.status == "boss_fight" else self.status
        return f"seed {self.seed}: {outcome}, {self.rows} rows, {self.seconds:.1f}s"


class Status:
    """Run totals and each worker's time since its last finished run, logged as a table every `interval` seconds."""

    HEADER_EVERY = 20

    def __init__(self, workers, interval):
        self.workers = workers
        self.runs = self.boss_fights = self.bosses_killed = self.rows = 0
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

    def record(self, fight):
        with self.lock:
            self.runs += 1
            self.boss_fights += fight.status == "boss_fight"
            self.bosses_killed += fight.won
            self.rows += fight.rows
            self.last_finished[current_thread().name] = time.monotonic()
        log.debug("%s", fight)

    def summary(self):
        with self.lock:
            return {"runs": self.runs, "boss_fights": self.boss_fights,
                    "bosses_killed": self.bosses_killed, "rows": self.rows}

    def header(self):
        workers = "".join(f"{f'w{i}':>5}" for i in range(self.workers))
        return f"{'runs':>7} {'boss kills':>11} {'rows':>10}  {workers}"

    def line(self):
        now = time.monotonic()
        with self.lock:
            kills = f"{self.bosses_killed}/{self.boss_fights}"
            since = [f"{now - last:>5.0f}" for last in self.last_finished.values()]
            since += [f"{'-':>5}"] * (self.workers - len(since))
            return f"{self.runs:>7,} {kills:>11} {self.rows:>10,}  {''.join(since)}"

    def report_every(self, interval):
        for count in itertools.count():
            if self.stopped.wait(interval):
                return
            if count % self.HEADER_EVERY == 0:
                log.info("%s", self.header())
            log.info("%s", self.line())


def to_parquet(msgpack_path, part, seconds):
    with msgpack_path.open("rb") as source:
        result = msgpack.unpack(source, raw=False)
    rows = [{**FIGHT, **row} for row in result["rows"]]
    if rows:
        pq.write_table(pa.Table.from_pylist(rows, schema=COMBAT_V2), part, compression="zstd")
    return Fight(result["seed"], result["status"], result.get("won", False), len(rows), seconds)


def fight(seed, binary, ascension, out, status):
    """One fight -> out/part-<seed>.parquet (no file if the run never reached the boss)."""
    with tempfile.TemporaryDirectory() as tmp:
        start = time.monotonic()
        subprocess.run([binary, str(seed), str(ascension), tmp], check=True)
        status.record(to_parquet(Path(tmp) / "fight.msgpack", out / f"part-{seed:06d}.parquet", time.monotonic() - start))


def generate(config, out):
    run = tomllib.loads(config.read_text())["run"]
    seeds = itertools.count() if run.get("forever") else range(run["fights"])
    workers = run["workers"]
    start = time.monotonic()
    log.info("starting %s seeds on %d workers -> %s", "unlimited" if run.get("forever") else run["fights"], workers, out)
    log.info("w0..w%d: seconds since that worker last finished a run", workers - 1)
    with Status(workers, run.get("status_seconds", 10)) as status, ThreadPoolExecutor(
            workers, thread_name_prefix="w", initializer=status.worker_started) as pool:
        play = partial(fight, binary=run["binary"], ascension=run.get("ascension", 1), out=out, status=status)
        for batch in batched(seeds, workers):
            for _ in pool.map(play, batch):
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
