"""Run files: frozen input copies, JSON lines and atomic JSON writes."""
import hashlib
import json
import os
from pathlib import Path
import shutil


def snapshot(path, out, name=None):
    """Read-only copy of `path` in out/, so a rebuild or retrain can't change this run; (copy, its sha256)."""
    copy = out / (name or Path(path).name)
    shutil.copy2(path, copy)
    copy.chmod(0o555 if os.access(path, os.X_OK) else 0o444)
    return copy.resolve(), hashlib.sha256(copy.read_bytes()).hexdigest()


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
