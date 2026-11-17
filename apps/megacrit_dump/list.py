"""Crawl the Mega Crit run-history Drive folder (recursing into subfolders) -> out/files.parquet + summary.json.

  PYTHONPATH=. .venv/bin/python -m runs.run megacrit_runs_v1 listing --no-compact -- \
      .venv/bin/python apps/megacrit_dump/list.py --out {out}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps.megacrit_dump import drive  # noqa: E402
from apps.megacrit_dump.schema import load  # noqa: E402

ROOT = "1c7MwTdLxnPgvmPbBEfNWa45YAUU53H0l"
TS = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})")


def parse_ts(name: str) -> dt.datetime | None:
    m = TS.match(name)
    try:
        return dt.datetime(*map(int, m.groups())) if m else None
    except ValueError:
        return None


def crawl(root: str, fetch_html=lambda fid: drive.fetch(drive.LIST_URL.format(fid)).decode("utf-8", "replace")):
    rows, folders = [], {}
    stack = [(root, "")]
    while stack:
        fid, path = stack.pop()
        entries = drive.parse_listing(fetch_html(fid))
        folders[path or "/"] = {"entries": len(entries), "capped": len(entries) >= drive.LIST_CAP}
        for eid, name, is_folder in entries:
            if is_folder:
                stack.append((eid, f"{path}/{name}" if path else name))
            else:
                rows.append({"file_id": eid, "name": name, "folder_path": path, "timestamp": parse_ts(name)})
    return rows, folders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, folders = crawl(a.root)
    rows.sort(key=lambda r: (r["timestamp"] is None, r["timestamp"], r["name"]))
    schema = load().FILES
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / "files.parquet", compression="zstd")
    ts = [r["timestamp"] for r in rows if r["timestamp"]]
    summary = {
        "files": len(rows),
        "first": min(ts).isoformat() if ts else None,
        "last": max(ts).isoformat() if ts else None,
        "folders": folders,
        "capped_folders": [p for p, f in folders.items() if f["capped"]],
        "complete": not any(f["capped"] for f in folders.values()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
