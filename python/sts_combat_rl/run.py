"""Run a job inside a fresh run directory. See runs/README.md."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS, SCRATCH = REPO / "runs", REPO / "scratch"
REPOS = {"sts_combat_rl": REPO, "sts_lightspeed": REPO.parent / "sts_lightspeed", "sts_ml": REPO.parent / "sts_ml"}
NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")

HELP = """\
Run a job in a new run directory:

  runs/schema=<schema>/date=<YYYY-MM-DD>/id=<id>/     (or scratch/... with --scratch)
    run.json   written by this launcher: status, command, git, inputs, summary
    logs/      stdout.log, stderr.log of the job
    out/       the job's outputs: part-<NNN>.parquet, checkpoint, episodes.jsonl, ...

The job gets its output dir as {out} in the command, or as $RUN_OUT (run dir: $RUN_DIR).
If the job writes out/summary.json, it is merged into run.json as "summary".
Exit code of the job is returned. Status ends as "done" (exit 0) or "failed".

examples:
  ./apps/bootstrap/run.sh apps/bootstrap/act1.toml [--scratch]
  PYTHONPATH=python .venv/bin/python -m sts_combat_rl.run value_net_v1 gen1-value \\
      --input combat_v2/2026-09-23/slime-gen1 -- python -m sts_combat_rl.training.train_value ...
  PYTHONPATH=python .venv/bin/python -m sts_combat_rl.run episodes_v1 quick-check --scratch -- ...

query with duckdb:
  SELECT * FROM read_parquet('runs/schema=combat_v3/*/*/out/*.parquet', hive_partitioning = true, union_by_name = true);
  SELECT * FROM read_json('runs/*/*/*/run.json');

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


def run_path(run_id: str) -> str:
    """'<schema>/<date>/<id>' -> 'schema=<schema>/date=<date>/id=<id>'."""
    schema, date, id_ = run_id.split("/")
    return f"schema={schema}/date={date}/id={id_}"


def run_dir(run_id: str) -> Path:
    """The directory of an existing run_id, in runs/ or scratch/."""
    for root in (RUNS, SCRATCH):
        if (root / run_path(run_id)).is_dir():
            return root / run_path(run_id)
    raise FileNotFoundError(f"no run {run_id!r} in runs/ or scratch/")


def bootstrap_inputs(run_id: str) -> list[str]:
    """The run's inputs that are bootstrap runs: schema combat_v3 with no inputs of their own."""
    inputs = json.loads((run_dir(run_id) / "run.json").read_text()).get("inputs") or []
    found = []
    for run in inputs:
        meta = json.loads((run_dir(run) / "run.json").read_text())
        if meta.get("schema") == "combat_v3" and not meta.get("inputs"):
            found.append(run)
    if not found:
        raise ValueError(f"{run_id}: no bootstrap combat_v3 input runs in {inputs}")
    return found


def run_parquet(run_id: str) -> list[Path]:
    """The parquet parts in a run's out/ (out/ also holds summary.json etc.)."""
    return sorted((run_dir(run_id) / "out").glob("*.parquet"))


def resolve_input(value: str) -> str:
    """A run_id (<schema>/<date>/<id>) or an existing path."""
    if value.count("/") == 2 and any((d / run_path(value)).is_dir() for d in (RUNS, SCRATCH)):
        return value
    if Path(value).exists():
        return str(Path(value).resolve())
    sys.exit(f"--input {value!r}: no such run_id in runs/ or scratch/, and no such path")


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sts_combat_rl.run", description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("schema", help="what the run produces, e.g. combat_v3, value_net_v1, episodes_v1 (runs/README.md)")
    parser.add_argument("id", help="short lowercase free text, e.g. slime-pbcs20k-gen1 ([a-z0-9_.-])")
    parser.add_argument("--input", action="append", default=[], metavar="RUN_ID|PATH", help="run this job reads from (repeatable); recorded as lineage")
    parser.add_argument("--scratch", action="store_true", help="smoke/preflight/probe: put in scratch/ (never queried, safe to delete)")
    parser.add_argument("--overwrite", action="store_true", help="delete an existing run with this id before starting")
    parser.add_argument("--live", action="store_true", help="also print job output to the terminal while retaining logs")
    parser.add_argument("--note", default="", help="free-text note stored in run.json")
    parser.usage = "%(prog)s SCHEMA ID [--input RUN_ID|PATH ...] [--scratch] [--overwrite] [--note TEXT] -- COMMAND..."
    argv = sys.argv[1:] if argv is None else argv
    split = argv.index("--") if "--" in argv else len(argv)
    args, cmd = parser.parse_args(argv[:split]), argv[split + 1:]
    if not cmd:
        parser.error("missing job command: put it after --")
    for field in ("schema", "id"):
        if not NAME.match(getattr(args, field)):
            parser.error(f"{field} {getattr(args, field)!r} must match {NAME.pattern}")
    inputs = [resolve_input(v) for v in args.input]

    run_id = f"{args.schema}/{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}/{args.id}"
    run_dir = (SCRATCH if args.scratch else RUNS) / run_path(run_id)
    if run_dir.exists():
        if not args.overwrite:
            sys.exit(f"{run_dir} already exists; pick another id or pass --overwrite")
        shutil.rmtree(run_dir)
    out, logs = run_dir / "out", run_dir / "logs"
    out.mkdir(parents=True)
    logs.mkdir()

    cmd = [c.replace("{out}", str(out)) for c in cmd]
    record = {
        "run_id": run_id,
        "schema": args.schema,
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
    env = {**os.environ, "RUN_ID": run_id, "RUN_DIR": str(run_dir), "RUN_OUT": str(out),
           "RUN_INTERACTIVE": "1" if args.live and sys.stdout.isatty() else "0"}
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

            terminal = [True]  # False once the terminal is gone; the log keeps going

            def relay(source, logfile):
                for line in source:
                    logfile.write(line)
                    logfile.flush()
                    with terminal_lock:
                        if not terminal[0]:
                            continue
                        try:
                            sys.stdout.buffer.write(line)
                            sys.stdout.buffer.flush()
                        except OSError:  # e.g. a closed pipe (`run ... | tail`)
                            terminal[0] = False

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
