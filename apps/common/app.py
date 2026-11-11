"""Config checks and run helpers shared by the apps."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(path, out, name=None):
    """Read-only copy of `path` in out/, so a rebuild or retrain can't change this run; (copy, its sha256)."""
    copy = out / (name or Path(path).name)
    shutil.copy2(path, copy)
    copy.chmod(0o555 if os.access(path, os.X_OK) else 0o444)
    return copy.resolve(), sha256(copy)


def read_jsonl(path, live=False):
    """The rows of a JSON-lines file, [] if it does not exist. `live`: its writer may be midway through the
    last line, which is then skipped; any other malformed line is an error."""
    path = Path(path)
    lines = [line for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if not (live and i == len(lines) - 1):
                raise
    return rows


def write_json(path, value):
    """Atomic: readers never see a half-written file."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, default=float) + "\n")
    os.replace(tmp, path)
