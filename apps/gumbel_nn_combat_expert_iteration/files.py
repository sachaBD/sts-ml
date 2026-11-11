"""JSON files the run reads and writes."""
import json
import os
from pathlib import Path


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
