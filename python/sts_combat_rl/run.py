"""Run a job inside a fresh run directory. See runs/README.md."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
REPOS = {"sts_combat_rl": REPO, "sts_lightspeed": REPO.parent / "sts_lightspeed", "sts_ml": REPO.parent / "sts_ml"}
KINDS = ("gen", "train", "eval")
NAME = re.compile(r"^[a-z0-9][a-z0-9.-]*$")

HELP = """\
Run a job in a new run directory:

  runs/run_id=<YYYY-MM-DD>_<kind>_<name>/     (or runs/scratch/... with --scratch)
    run.json   written by this launcher: status, command, git, inputs, summary
    logs/      stdout.log, stderr.log of the job
    out/       the job's outputs: parquet under out/schema=<schema>/..., checkpoint, episodes.jsonl, ...

The job gets its output dir as {out} in the command, or as $RUN_OUT (run dir: $RUN_DIR).
If the job writes out/summary.json, it is merged into run.json as "summary".
Exit code of the job is returned. Status ends as "done" (exit 0) or "failed".

examples:
  ./apps/bootstrap/run.sh apps/bootstrap/slime.toml [--scratch]
  PYTHONPATH=python .venv/bin/python -m sts_combat_rl.run train gen1-value \\
      --input 2026-09-23_gen_slime-pbcs20k-gen1 -- python -m sts_combat_rl.training.train_value ...
  PYTHONPATH=python .venv/bin/python -m sts_combat_rl.run eval quick-check --scratch -- ...

query with duckdb:
  SELECT * FROM read_parquet('runs/run_id=*/out/schema=combat_v1/**/*.parquet', hive_partitioning = true, union_by_name = true);
  SELECT * FROM read_json('runs/run_id=*/run.json');

full contract for jobs, schemas and columns: runs/README.md
"""


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git_state() -> dict:
    state = {}
    for name, path in REPOS.items():
        if not (path / ".git").exists():
            continue
        git = ["git", "-C", str(path)]
        rev = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
        state[name] = {"rev": rev, "dirty": dirty}
    return state


def resolve_input(value: str) -> str:
    """A run_id (with or without the 'run_id=' prefix) or an existing path."""
    run_id = value.removeprefix("run_id=")
    if any((d / f"run_id={run_id}").is_dir() for d in (RUNS, RUNS / "scratch")):
        return run_id
    if Path(value).exists():
        return str(Path(value).resolve())
    sys.exit(f"--input {value!r}: no such run_id in runs/ or runs/scratch/, and no such path")


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sts_combat_rl.run", description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=KINDS, help="what the job produces: gen=training data, train=checkpoint, eval=episodes/metrics")
    parser.add_argument("name", help="short lowercase name, e.g. slime-pbcs20k-gen1 ([a-z0-9.-])")
    parser.add_argument("--input", action="append", default=[], metavar="RUN_ID|PATH", help="run this job reads from (repeatable); recorded as lineage")
    parser.add_argument("--scratch", action="store_true", help="smoke/preflight/probe: put in runs/scratch/ (excluded from normal queries, safe to delete)")
    parser.add_argument("--live", action="store_true", help="also print job output to the terminal while retaining logs")
    parser.add_argument("--note", default="", help="free-text note stored in run.json")
    parser.usage = "%(prog)s {gen,train,eval} NAME [--input RUN_ID|PATH ...] [--scratch] [--note TEXT] -- COMMAND..."
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    args, cmd = parser.parse_args(argv[:split]), argv[split + 1:]
    if not cmd:
        parser.error("missing job command: put it after --")
    if not NAME.match(args.name):
        parser.error(f"name {args.name!r} must match {NAME.pattern}")
    inputs = [resolve_input(v) for v in args.input]

    run_id = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}_{args.kind}_{args.name}"
    run_dir = (RUNS / "scratch" if args.scratch else RUNS) / f"run_id={run_id}"
    if run_dir.exists():
        sys.exit(f"{run_dir} already exists; pick another name (runs are never overwritten)")
    out, logs = run_dir / "out", run_dir / "logs"
    out.mkdir(parents=True)
    logs.mkdir()

    cmd = [c.replace("{out}", str(out)) for c in cmd]
    record = {
        "run_id": run_id,
        "kind": args.kind,
        "scratch": args.scratch,
        "status": "running",
        "note": args.note,
        "started": now(),
        "finished": None,
        "exit_code": None,
        "host": socket.gethostname(),
        "pid": None,
        "cwd": os.getcwd(),
        "command": cmd,
        "inputs": inputs,
        "git": git_state(),
        "summary": None,
    }
    env = {**os.environ, "RUN_ID": run_id, "RUN_DIR": str(run_dir), "RUN_OUT": str(out)}
    print(f"run: {run_dir}", file=sys.stderr)
    print(f"log: {logs / 'stdout.log'} (stderr: {logs / 'stderr.log'})", file=sys.stderr)

    with open(logs / "stdout.log", "wb") as so, open(logs / "stderr.log", "wb") as se:
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE if args.live else so,
                                    stderr=subprocess.PIPE if args.live else se, env=env)
        except OSError as e:
            record.update(status="failed", finished=now(), exit_code=127, summary={"error": str(e)})
            write_json(run_dir / "run.json", record)
            sys.exit(f"run: failed to start job: {e} ({run_dir})")
        record["pid"] = proc.pid
        write_json(run_dir / "run.json", record)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda s, _f: proc.send_signal(s))
        if args.live:
            terminal_lock = threading.Lock()

            def relay(source, logfile):
                for line in source:
                    logfile.write(line)
                    logfile.flush()
                    with terminal_lock:
                        sys.stdout.buffer.write(line)
                        sys.stdout.buffer.flush()

            relays = [threading.Thread(target=relay, args=(proc.stdout, so)),
                      threading.Thread(target=relay, args=(proc.stderr, se))]
            for thread in relays:
                thread.start()
        code = proc.wait()
        if args.live:
            for thread in relays:
                thread.join()

    summary = out / "summary.json"
    if summary.exists():
        try:
            record["summary"] = json.loads(summary.read_text())
        except json.JSONDecodeError as e:
            record["summary"] = {"error": f"bad summary.json: {e}"}
    record.update(status="done" if code == 0 else "failed", finished=now(), exit_code=code)
    write_json(run_dir / "run.json", record)
    print(f"run: {record['status']} (exit {code}) {run_dir}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
