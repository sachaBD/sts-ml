#!/usr/bin/env python3
"""Bootstrap combat data generation: teacher-search fights, written continuously.

Runs N worker processes (build/bootstrap_fight_worker), feeds them one fight per job,
and writes finished fights to parquet shards that are closed and renamed into place as
they fill. Killing this process (Ctrl+C / SIGTERM) is the intended way to stop: workers
finish the fight in hand, shards flush, and everything already on disk stays valid.

Output contract: runs/README.md. Launch through the run launcher, e.g.

    ./apps/bootstrap/run.sh slime-bootstrap --workers 12 --forever
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import msgpack
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = REPO.parent / "sts_ml" / "runs" / "slime-entry-natural-220.jsonl"
TEACHER = "sts_ml PublicBeliefCombatSearch, rollout mode 2 (guided), objective mode 0"
GOLDEN_GAMMA = 0x9E3779B97F4A7C15

# schema=combat_v1 (runs/README.md). Columns only; run_id/schema/partition keys live in the path.
F32 = pa.float32()
COMBAT_V1 = pa.schema(
    [
        ("episode_id", pa.int64()),
        ("decision_index", pa.int32()),
        ("turn", pa.int32()),
        ("entry_id", pa.string()),
        ("deck_signature", pa.string()),
        ("combat_seed", pa.uint64()),
        ("starting_hp", pa.int16()),
        ("starting_max_hp", pa.int16()),
        ("encoding_version", pa.int32()),
        ("global_numeric", pa.list_(F32, 50)),
        ("cards", pa.list_(pa.struct([("card_id", pa.int16()), ("zone", pa.int8()), ("card_type", pa.int8()),
                                      ("target_type", pa.int8()), ("numeric", pa.list_(F32, 14))]))),
        ("monsters", pa.list_(pa.struct([("monster_id", pa.int16()), ("move_id", pa.int16()),
                                         ("numeric", pa.list_(F32, 9))]))),
        ("card_monster_interactions", pa.list_(pa.struct([("card_index", pa.int16()), ("monster_index", pa.int8()),
                                                          ("numeric", pa.list_(F32, 6))]))),
        ("input_state", pa.int16()),
        ("card_selection_task", pa.int16()),
        ("actions", pa.list_(pa.struct([("action", pa.int32()), ("description", pa.string()),
                                        ("visits", pa.int64()), ("mean_value", F32)]))),
        ("chosen_action", pa.int32()),
        ("was_random", pa.bool_()),
        ("root_value", F32),
        ("won", pa.bool_()),
        ("final_hp", pa.int16()),
        ("potions", pa.int8()),
        ("terminal_value", F32),
        ("row_kind", pa.string()),
        ("parent_action", pa.int32()),
        ("simulations_used", pa.int64()),
    ]
)
SCHEMAS = {"combat_v1": COMBAT_V1}


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


class Shards:
    """Rolling parquet parts. Only complete files appear under the final name."""

    def __init__(self, directory: Path, schema: pa.Schema, max_rows: int, max_seconds: float):
        self.dir, self.schema = directory, schema
        self.max_rows, self.max_seconds = max_rows, max_seconds
        self.lock = threading.Lock()
        self.counter = itertools.count()
        self.written = 0  # completed parts
        self.rows = 0  # rows in completed parts
        directory.mkdir(parents=True, exist_ok=True)

    def open_part(self) -> tuple[Path, Path]:
        with self.lock:
            index = next(self.counter)
        final = self.dir / f"part-{index:03d}.parquet"
        return self.dir / f".part-{index:03d}.parquet.tmp", final

    def finish(self, rows: int) -> None:
        with self.lock:
            self.written += 1
            self.rows += rows


class ShardWriter:
    """One worker's slice of the output: buffer fights, flush a closed part periodically."""

    def __init__(self, shards: Shards):
        self.shards = shards
        self.batch: list[dict] = []
        self.opened = time.monotonic()

    def add(self, rows: list[dict]) -> None:
        self.batch.extend(rows)
        if len(self.batch) >= self.shards.max_rows or time.monotonic() - self.opened > self.shards.max_seconds:
            self.flush()

    def flush(self) -> None:
        if not self.batch:
            self.opened = time.monotonic()
            return
        tmp, final = self.shards.open_part()
        table = pa.Table.from_pylist(self.batch, schema=self.shards.schema)
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(final)  # atomic: only valid, closed parquet is ever visible
        self.shards.finish(len(self.batch))
        self.batch, self.opened = [], time.monotonic()


class Roots:
    """The pool of Slime-entry roots to fight from. Grows if a seed exporter is supplied."""

    def __init__(self, entries: list[dict]):
        self.entries = entries
        self.lock = threading.Lock()
        self.exported = len(entries)

    def __len__(self) -> int:
        with self.lock:
            return len(self.entries)

    def get(self, index: int) -> dict | None:
        with self.lock:
            return self.entries[index % len(self.entries)] if self.entries else None

    def extend(self, entries: list[dict]) -> None:
        with self.lock:
            self.entries.extend(entries)


def load_roots(path: Path) -> list[dict]:
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [e for e in entries if e.get("status") == "accepted"]


def export_seeds(command: str, start: int, count: int, scratch: Path) -> list[dict]:
    """Run the configured seed exporter for [start, start+count) and return accepted entries."""
    target = scratch / f"seeds-{start}.jsonl"
    cmd = command.format(start=start, stop=start + count, out=str(target))
    subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
    return load_roots(target) if target.exists() else []


class Status:
    """Per-worker progress, from the workers' stderr heartbeats."""

    def __init__(self, workers: int, stall_seconds: float):
        self.stall_seconds = stall_seconds
        self.lock = threading.Lock()
        self.state = [{"episode": None, "decision": 0, "turn": 0, "since": time.monotonic(), "fights": 0,
                       "busy": False} for _ in range(workers)]
        self.fights = self.wins = self.rows = 0
        self.durations: list[float] = []

    def heartbeat(self, worker: int, fields: dict[str, str]) -> None:
        with self.lock:
            self.state[worker].update(episode=fields.get("episode"), decision=int(fields.get("decision", 0)),
                                      turn=int(fields.get("turn", 0)), since=time.monotonic(), busy=True)

    def finished(self, worker: int, fields: dict[str, str]) -> None:
        with self.lock:
            self.fights += 1
            self.wins += fields.get("won") == "1"
            self.rows += int(fields.get("rows", 0))
            self.state[worker]["fights"] += 1
            self.state[worker].update(since=time.monotonic(), decision=0, turn=0, busy=False)
            self.durations.append(float(fields.get("secs", 0)))
            del self.durations[:-200]

    def report(self, worker: int) -> str:
        with self.lock:
            state, waited = self.state[worker], time.monotonic() - self.state[worker]["since"]
            median = sorted(self.durations)[len(self.durations) // 2] if self.durations else 0.0
        if not state["busy"]:
            flag = "idle"
        elif waited > self.stall_seconds:
            flag = "STALLED"
        else:
            flag = "SLOW" if median and waited > 3 * median else "ok"
        episode = state["episode"] if state["episode"] is not None else "-"
        return (f"  w{worker:02d} ep {episode:>6} dec {state['decision']:>3} turn {state['turn']:>2} "
                f"{waited:6.1f}s  {flag}   ({state['fights']} fights)")


def run_worker(index: int, command: list[str], jobs: queue.Queue, shards: Shards, status: Status,
               stop: threading.Event, drained: threading.Event, depth: int, errors: list) -> None:
    """One generator process: feed it jobs, read finished fights, write shards."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0)
    writer = ShardWriter(shards)
    slots = threading.Semaphore(depth)  # bound in-flight jobs so a stop wastes at most `depth` fights

    def feed() -> None:
        try:
            while not stop.is_set():
                slots.acquire()
                if stop.is_set():
                    break
                try:
                    job = jobs.get(timeout=0.5)
                except queue.Empty:
                    slots.release()
                    if drained.is_set():
                        break
                    continue
                process.stdin.write((json.dumps(job) + "\n").encode())
        except (BrokenPipeError, ValueError):
            pass
        finally:
            try:
                process.stdin.close()  # EOF: the worker exits after its current fight
            except (BrokenPipeError, ValueError):
                pass

    def watch() -> None:
        for raw in process.stderr:
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            kind, _, rest = line.partition(" ")
            fields = dict(part.split("=", 1) for part in rest.split() if "=" in part)
            if kind == "hb":
                status.heartbeat(index, fields)
            elif kind == "fight":
                status.finished(index, fields)
                slots.release()
                log(f"w{index:02d} fight {line[len('fight '):]}")
            else:
                log(f"w{index:02d} {line}")

    feeder = threading.Thread(target=feed, daemon=True)
    watcher = threading.Thread(target=watch, daemon=True)
    feeder.start()
    watcher.start()
    try:
        unpacker = msgpack.Unpacker(raw=False)
        while chunk := process.stdout.read(1 << 16):  # unbuffered: returns as soon as bytes arrive
            unpacker.feed(chunk)
            for fight in unpacker:  # one object per finished fight
                writer.add(fight["rows"])
        writer.flush()
        code = process.wait()
        if code:
            errors.append(RuntimeError(f"worker {index} exited {code}"))
    except BaseException as error:  # noqa: BLE001 - reported by the caller
        errors.append(error)
        process.kill()
    finally:
        writer.flush()
        stop.set()
        slots.release()
        feeder.join(timeout=5)
        watcher.join(timeout=5)


def produce(jobs: queue.Queue, roots: Roots, args, stop: threading.Event, drained: threading.Event,
            deadline: float | None, scratch: Path) -> None:
    """Enqueue one job per fight: each root is replayed `--replicates` times, then the next root."""
    episode, index, exporter = 0, 0, args.export_cmd
    next_seed = args.export_start
    while not stop.is_set():
        if args.fights and episode >= args.fights:
            break
        if deadline and time.monotonic() > deadline:
            break
        if exporter and index // max(1, args.replicates) >= len(roots) - args.export_ahead:
            try:
                new = export_seeds(exporter, next_seed, args.export_batch, scratch)
                next_seed += args.export_batch
                roots.extend(new)
                log(f"seeds: exported {args.export_batch}, accepted {len(new)}, pool {len(roots)}")
            except subprocess.CalledProcessError as error:
                log(f"seeds: exporter failed ({error}); continuing with pool of {len(roots)}")
                exporter = None
        pool = len(roots)
        if not pool:
            log("no roots available; stopping")
            break
        # index -> (root, replicate): R fights per root, then the next root; each full pass
        # over the pool moves on to fresh replicates, so seeds never repeat.
        cycle, position = divmod(index, args.replicates * pool)
        entry = roots.get(position // args.replicates)
        replicate = position % args.replicates + cycle * args.replicates
        seed = (int(entry["seed"]) ^ (GOLDEN_GAMMA * (replicate + 1))) % 2**64
        job = {"episode_id": episode, "combat_seed": seed, "entry": entry}
        while not stop.is_set():
            try:
                jobs.put(job, timeout=0.5)
                break
            except queue.Full:
                continue
        else:
            break
        episode, index = episode + 1, index + 1
    drained.set()
    log(f"producer done: {episode} fights queued")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output dir from the run launcher ({out} / $RUN_OUT)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--forever", action="store_true", help="generate until killed")
    parser.add_argument("--fights", type=int, default=0, help="stop after N fights")
    parser.add_argument("--minutes", type=float, default=0.0, help="stop after N minutes")
    parser.add_argument("--replicates", type=int, default=1, help="fights per root before moving to the next")
    parser.add_argument("--roots", type=Path, default=DEFAULT_ROOTS, help="accepted Slime-entry JSONL to start from")
    parser.add_argument("--simulations", type=int, default=15000)
    parser.add_argument("--max-actions", type=int, default=512)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--random-window", type=int, default=24, help="one random move per fight at decision U[0,N); 0 disables")
    parser.add_argument("--child-min-visits", type=int, default=50, help="child rows for tried-but-unplayed moves; 0 disables")
    parser.add_argument("--worker-binary", type=Path, default=REPO / "build" / "bootstrap_fight_worker")
    parser.add_argument("--schema", default="combat_v1", choices=sorted(SCHEMAS))
    parser.add_argument("--partition", default="act=1/floor=16/encounter=slime_boss", help="hive path under schema=<name>/")
    parser.add_argument("--shard-rows", type=int, default=20000, help="roll a part after this many rows")
    parser.add_argument("--shard-seconds", type=float, default=300.0, help="roll a part after this long")
    parser.add_argument("--status-seconds", type=float, default=15.0)
    parser.add_argument("--stall-seconds", type=float, default=120.0, help="flag a worker with no heartbeat for this long")
    parser.add_argument("--in-flight", type=int, default=2, help="jobs per worker held in the pipe")
    parser.add_argument("--export-cmd", default="", help="shell command producing new roots, with {start} {stop} {out}")
    parser.add_argument("--export-start", type=int, default=0, help="first seed for --export-cmd")
    parser.add_argument("--export-batch", type=int, default=32, help="seeds per exporter call")
    parser.add_argument("--export-ahead", type=int, default=64, help="keep this many unplayed roots in the pool")
    args = parser.parse_args(argv)
    if not (args.forever or args.fights or args.minutes):
        parser.error("pick a stopping rule: --forever, --fights N or --minutes M")

    entries = load_roots(args.roots) if args.roots.exists() else []
    if not entries and not args.export_cmd:
        parser.error(f"no accepted roots in {args.roots} and no --export-cmd")
    # `--replicates R`: R fights per root, then the next root, so index//R selects the root.
    roots = Roots(entries)
    shards = Shards(args.out / f"schema={args.schema}" / args.partition, SCHEMAS[args.schema],
                    args.shard_rows, args.shard_seconds)
    scratch = args.out / "seeds"
    if args.export_cmd:
        scratch.mkdir(parents=True, exist_ok=True)
    status = Status(args.workers, args.stall_seconds)
    stop, drained, errors = threading.Event(), threading.Event(), []
    jobs: queue.Queue = queue.Queue(maxsize=args.workers * 2)
    command = [str(args.worker_binary), "--simulations", str(args.simulations), "--max-actions", str(args.max_actions),
               "--particles", str(args.particles), "--random-window", str(args.random_window),
               "--child-min-visits", str(args.child_min_visits)]

    hits = {"count": 0}

    def handle(signum, _frame):
        hits["count"] += 1
        stop.set()
        drained.set()
        if hits["count"] == 1:
            log(f"signal {signal.Signals(signum).name}: finishing fights in flight, then flushing (again to force)")
        else:
            log("second signal: forcing shutdown")
            raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle)

    start = time.monotonic()
    deadline = start + args.minutes * 60 if args.minutes else None
    log(f"out {shards.dir}")
    log(f"teacher {args.simulations} sims, {args.particles} particles | {args.workers} workers | "
        f"{len(roots)} roots x {args.replicates} replicates | "
        f"stop: {'never' if args.forever else (f'{args.fights} fights' if args.fights else f'{args.minutes} min')}")

    threads = [threading.Thread(target=run_worker,
                                args=(i, command, jobs, shards, status, stop, drained, args.in_flight, errors),
                                daemon=True)
               for i in range(args.workers)]
    for thread in threads:
        thread.start()
    producer = threading.Thread(target=produce, args=(jobs, roots, args, stop, drained, deadline, scratch),
                                daemon=True)
    producer.start()

    try:
        while any(t.is_alive() for t in threads):
            time.sleep(args.status_seconds)
            elapsed = (time.monotonic() - start) / 60
            rate = status.fights / elapsed if elapsed else 0
            wins = 100 * status.wins / status.fights if status.fights else 0
            log(f"up {elapsed:.1f}m | {sum(t.is_alive() for t in threads)}/{args.workers} workers | "
                f"{status.fights} fights ({rate:.1f}/min) | {status.rows} rows | {wins:.0f}% teacher wins | "
                f"roots {len(roots)} | shards {shards.written} ({shards.rows} rows on disk)")
            for index in range(args.workers):
                print(status.report(index), flush=True)

    except KeyboardInterrupt:
        stop.set()
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=600)

    elapsed = time.monotonic() - start
    summary = {
        "teacher": TEACHER,
        "simulations": args.simulations,
        "particles": args.particles,
        "max_actions": args.max_actions,
        "early_stop": "forced moves get 500 simulations; otherwise stop once the top root move can't be overtaken",
        "random_move": f"one per fight, uniformly random legal action at decision U[0, {args.random_window})",
        "child_rows": f"non-chosen root moves with >= {args.child_min_visits} visits; label = teacher mean value (root_value)",
        "root_source": str(args.roots.resolve()),
        "root_source_sha256": hashlib.sha256(args.roots.read_bytes()).hexdigest() if args.roots.exists() else None,
        "roots": len(roots),
        "replicates": args.replicates,
        "workers": args.workers,
        "seeds": {
            "combat_seed": "entry seed xor 0x9e3779b97f4a7c15*(replicate+1)",
            "random_move_rng": "mt19937_64(combat_seed xor 0xe9510)",
            "search": "particle seeds from PublicBeliefCombatSearch::publicObservation",
        },
        "terminal_value": "win: (35 + final_hp + 4*potions) / (55 + max_hp); loss: 0",
        "episodes": status.fights,
        "wins": status.wins,
        "rows": shards.rows,
        "parts": shards.written,
        "wall_seconds": round(elapsed, 1),
        "stopped_by": "signal" if hits["count"] else "limit",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"done: {status.fights} fights, {shards.rows} rows in {shards.written} parts, "
        f"{elapsed / 60:.1f} min -> {shards.dir}")
    if errors and not hits["count"]:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
