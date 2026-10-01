"""The app entry point, config checks and run helpers shared by the apps."""
import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

from sts_combat_rl.run import run_dir

ORACLE_BANNER = ("\n" + "!" * 78 + "\n!!  ORACLE MODE: the teacher searches the TRUE state (perfect RNG / draw-order\n"
                 "!!  foresight). Upper-bound results, not fair play. Rows are tagged oracle = true.\n" + "!" * 78)
FAIR_PLAY = "oracle: off (teacher searches sampled beliefs, fair play)"


def main(run, inputs=lambda config: ()):
    """Command line of every app (apps/common/launch.sh):
      SCRIPT CONFIG.toml --out DIR   run(config, config_path, out); its return value is the exit code
      SCRIPT CONFIG.toml --inputs    print inputs(config), the run ids it reads, one per line
    """
    parser = argparse.ArgumentParser(description=sys.modules["__main__"].__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, help="run output dir")
    parser.add_argument("--inputs", action="store_true", help="print the input run ids (for apps/common/launch.sh)")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if args.inputs:
        print(*inputs(config), sep="\n")
        return
    if args.out is None:
        parser.error("--out is required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    sys.exit(run(config, args.config, args.out) or 0)


def check_keys(table, allowed, name):
    """Exit on a config key outside `allowed` (a typo would otherwise be ignored)."""
    if unknown := sorted(set(table) - set(allowed)):
        sys.exit(f"unknown [{name}] keys: {unknown}")


def required(table, key, name, kind=None):
    """table[key], of type `kind` if given (int excludes bool; float accepts int); exits naming the missing or
    mistyped [name].key. Settings have no defaults."""
    if key not in table:
        sys.exit(f"missing [{name}].{key}")
    value = table[key]
    if kind is not None:
        kinds = (int, float) if kind is float else kind
        if not isinstance(value, kinds) or (isinstance(value, bool) and kind is not bool):
            sys.exit(f"[{name}].{key} must be {kind.__name__}, got {value!r}")
    return value


def exactly_when(table, key, applies, when, name):
    """Exit unless [name].key is present exactly when it applies (settings that only sometimes apply)."""
    if applies != (key in table):
        sys.exit(f"[{name}].{key} is required {when}, and only then")


TEACHER_KEYS = {"leaf", "simulations", "oracle", "random_move", "particles", "rollout_turns", "rollout_steps",
                "merge_identical_cards", "stop_factor", "c_puct", "fpu_reduction", "prior_floor", "search_salt",
                "tree_reuse"}
POLICY_KEYS = ("c_puct", "fpu_reduction", "prior_floor")  # leaf policy_net: required, and only then


def teacher_settings(run):
    """The workers' teacher request (apps/common/teacher_request.hpp) from [run]: leaf (guided_rollout, value_net,
    hybrid or policy_net), simulations, oracle, random_move; particles unless oracle; rollout_turns / rollout_steps
    with leaf hybrid only; c_puct / fpu_reduction / prior_floor with leaf policy_net only; opt-in search tweaks
    merge_identical_cards / stop_factor / search_salt / tree_reuse. The worker validates the values."""
    teacher = {"leaf": required(run, "leaf", "run", str), "simulations": required(run, "simulations", "run", int),
               "oracle": required(run, "oracle", "run", bool), "random_move": required(run, "random_move", "run", bool)}
    exactly_when(run, "particles", not teacher["oracle"], "unless oracle", "run")
    for key in ("rollout_turns", "rollout_steps"):
        exactly_when(run, key, teacher["leaf"] == "hybrid", "with leaf hybrid", "run")
    for key in POLICY_KEYS:
        exactly_when(run, key, teacher["leaf"] == "policy_net", "with leaf policy_net", "run")
    return {**teacher, **{k: run[k] for k in TEACHER_KEYS - set(teacher) if k in run}}


def run_json(run_id):
    return json.loads((run_dir(run_id) / "run.json").read_text())


def value_run(run_id):
    """A finished value_net_v1 run: its out/ files (checkpoint, checkpoint_json, weights) and checkpoint metadata."""
    record = run_json(run_id)
    if record["status"] != "done":
        sys.exit(f"{run_id}: status {record['status']!r}, need done")
    out, summary = run_dir(run_id) / "out", record["summary"]
    files = {k: out / summary[k] for k in ("checkpoint", "checkpoint_json", "weights")}
    return SimpleNamespace(run_id=run_id, **files, meta=json.loads(files["checkpoint_json"].read_text()))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(path, out, name=None):
    """Read-only copy of `path` in out/, so a rebuild or retrain can't change this run; (copy, its sha256)."""
    copy = out / (name or Path(path).name)
    shutil.copy2(path, copy)
    copy.chmod(0o555 if os.access(path, os.X_OK) else 0o444)
    return copy.resolve(), sha256(copy)


def write_json(path, value):
    """Atomic: readers never see a half-written file."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    os.replace(tmp, path)
