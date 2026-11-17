"""Pull filtered runs from listed Drive files into Parquet. Raw files are never written to disk.

  PYTHONPATH=. .venv/bin/python -m runs.run megacrit_runs_v1 <id> --no-compact -- \
      .venv/bin/python apps/megacrit_dump/pull.py --files <files.parquet> --out {out} --n-files 50 \
      --filter-character IRONCLAD --filter-ascension 20

Resumable: out/done/ records processed file_ids (written together with the rows they produced). Stops with exit 3
(and a summary) if Drive throttles us.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps.megacrit_dump import drive  # noqa: E402
from apps.megacrit_dump.schema import load  # noqa: E402

CHAMP = ("Champ", "The Champ")  # the dump uses "Champ"
FLUSH_RUNS = 10_000


def _int(v):
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def at(lst, i):
    return _int(lst[i]) if isinstance(lst, list) and 0 <= i < len(lst) else None


def champ_fields(ev: dict) -> dict:
    """Champ-fight columns. damage_taken[].floor is the floor F of the fight; *_per_floor[i] is the value at the end of
    floor i+1, so the state entering floor F is index F-2."""
    out = dict.fromkeys(("champ_floor", "champ_damage", "champ_turns", "hp_before_champ", "max_hp_before_champ"))
    out.update(reached_champ=False, champ_won=False)
    for d in ev.get("damage_taken") or []:
        if isinstance(d, dict) and d.get("enemies") in CHAMP:
            floor = _int(d.get("floor"))
            out.update(reached_champ=True, champ_floor=floor, champ_damage=_int(d.get("damage")),
                       champ_turns=_int(d.get("turns")))
            if floor is not None:
                out["hp_before_champ"] = at(ev.get("current_hp_per_floor"), floor - 2)
                out["max_hp_before_champ"] = at(ev.get("max_hp_per_floor"), floor - 2)
                fr = _int(ev.get("floor_reached"))
                out["champ_won"] = bool(ev.get("victory")) or (fr is not None and fr > floor)
            break
    return out


def keep(ev: dict, character: str, ascension: int) -> bool:
    return (ev.get("character_chosen") == character and _int(ev.get("ascension_level")) == ascension
            and not (ev.get("is_daily") or ev.get("is_endless") or ev.get("is_trial") or ev.get("chose_seed")))


def process(data: bytes, file_id: str, name: str, character: str, ascension: int):
    """Gunzip+parse one file -> (kept rows, runs_seen). Everything not kept is dropped here."""
    items = json.loads(gzip.decompress(data))
    rows, seen = [], 0
    for item in items:
        ev = item.get("event", item) if isinstance(item, dict) else None
        if not isinstance(ev, dict):
            continue
        seen += 1
        if not keep(ev, character, ascension) or ev.get("play_id") is None:
            continue
        rows.append({
            "play_id": str(ev["play_id"]), "source_file_id": file_id, "source_name": name,
            "timestamp": _int(ev.get("timestamp")), "build_version": ev.get("build_version"),
            "character": ev.get("character_chosen"), "ascension": _int(ev.get("ascension_level")),
            "victory": bool(ev.get("victory")), "floor_reached": _int(ev.get("floor_reached")),
            "seed_played": None if ev.get("seed_played") is None else str(ev["seed_played"]),
            **champ_fields(ev), "raw": json.dumps(ev, separators=(",", ":")),
        })
    return rows, seen


def exclude_done(table: pa.Table, runs: list[str]) -> pa.Table:
    """Drop listed files that earlier pull runs already processed (their out/done/ file_ids)."""
    import pyarrow.compute as pc
    done = set()
    for r in runs:
        r = Path(r)
        d = r / "out" / "done" if (r / "out" / "done").is_dir() else r / "done"
        for p in d.glob("part-*.parquet"):
            done.update(pq.read_table(p, columns=["file_id"]).column(0).to_pylist())
    return table.filter(pc.invert(pc.is_in(table["file_id"], value_set=pa.array(sorted(done), pa.string()))))


def choose_files(table: pa.Table, n: int | None) -> list[dict]:
    """Deterministic: n targets evenly spaced in time across [first, last]; nearest unused file to each."""
    rows = [r for r in table.to_pylist() if r["timestamp"] is not None]
    rows.sort(key=lambda r: (r["timestamp"], r["name"]))
    if n is None or n >= len(rows):
        return rows
    if n <= 0:
        return []
    t0, t1 = rows[0]["timestamp"].timestamp(), rows[-1]["timestamp"].timestamp()
    ts = [r["timestamp"].timestamp() for r in rows]
    import bisect
    used, picked = set(), []
    for k in range(n):
        target = t0 + (t1 - t0) * (k / (n - 1) if n > 1 else 0)
        j = bisect.bisect_left(ts, target)
        lo, hi = j - 1, j
        while True:  # expand outward to the nearest unused index
            cands = [i for i in (lo, hi) if 0 <= i < len(rows) and i not in used]
            if cands:
                i = min(cands, key=lambda i: abs(ts[i] - target))
                break
            lo, hi = lo - 1, hi + 1
        used.add(i)
        picked.append(i)
    return [rows[i] for i in sorted(picked)]


class Sink:
    """Buffers kept rows + done records; writes a runs part and a done part together (done only after rows)."""

    def __init__(self, out: Path, schema):
        self.runs_dir, self.done_dir = out / "runs", out / "done"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.done_dir.mkdir(parents=True, exist_ok=True)
        self.schema, self.rows, self.done = schema, [], []
        self.seen_ids: set[str] = set()
        self.done_ids: set[str] = set()
        for p in self.runs_dir.glob("part-*.parquet"):
            self.seen_ids.update(pq.read_table(p, columns=["play_id"]).column(0).to_pylist())
        for p in self.done_dir.glob("part-*.parquet"):
            self.done_ids.update(pq.read_table(p, columns=["file_id"]).column(0).to_pylist())
        self.n_parts = len(list(self.done_dir.glob("part-*.parquet")))

    def add(self, file_id: str, rows: list[dict], seen: int, nbytes: int):
        fresh = []
        for r in rows:
            if r["play_id"] not in self.seen_ids:
                self.seen_ids.add(r["play_id"])
                fresh.append(r)
        self.rows += fresh
        self.done.append({"file_id": file_id, "runs_seen": seen, "runs_kept": len(fresh),
                          "kept_reached_champ": sum(r["reached_champ"] for r in fresh),
                          "kept_champ_won": sum(r["champ_won"] for r in fresh), "bytes": nbytes})
        if len(self.rows) >= FLUSH_RUNS:
            self.flush()

    def flush(self):
        if not self.done:
            return
        n = self.n_parts
        S = self.schema
        if self.rows:
            pq.write_table(pa.Table.from_pylist(self.rows, schema=S.RUNS), self.runs_dir / f"part-{n:05d}.parquet",
                           compression="zstd")
        pq.write_table(pa.Table.from_pylist(self.done, schema=S.DONE), self.done_dir / f"part-{n:05d}.parquet",
                       compression="zstd")
        self.done_ids.update(d["file_id"] for d in self.done)
        self.rows, self.done, self.n_parts = [], [], n + 1


def totals(done_dir: Path) -> dict:
    t = dict(files_processed=0, runs_seen=0, runs_kept=0, kept_reached_champ=0, kept_champ_wins=0, bytes_downloaded=0)
    for p in done_dir.glob("part-*.parquet"):
        d = pq.read_table(p).to_pydict()
        t["files_processed"] += len(d["file_id"])
        t["runs_seen"] += sum(d["runs_seen"])
        t["runs_kept"] += sum(d["runs_kept"])
        t["kept_reached_champ"] += sum(d["kept_reached_champ"])
        t["kept_champ_wins"] += sum(d["kept_champ_won"])
        t["bytes_downloaded"] += sum(d["bytes"])
    return t


def run(files: list[dict], out: Path, character: str, ascension: int, workers: int, download=drive.download,
        log=print) -> dict:
    S = load()
    sink = Sink(out, S)
    todo = [f for f in files if f["file_id"] not in sink.done_ids]
    t0 = time.time()
    stop = threading.Event()
    status = {"throttled": False, "errors": []}

    def work(f):
        if stop.is_set():
            return None
        try:
            data = download(f["file_id"])
            rows, seen = process(data, f["file_id"], f["name"], character, ascension)
            return f, rows, seen, len(data)
        except drive.Throttled as e:
            stop.set()
            status["throttled"] = True
            status["errors"].append(f"{f['name']}: {e}")
        except Exception as e:  # one bad file must not kill the pull; it is retried on the next start
            status["errors"].append(f"{f['name']}: {type(e).__name__}: {e}")
        return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(work, todo), 1):
            if res:
                f, rows, seen, nbytes = res
                sink.add(f["file_id"], rows, seen, nbytes)
            if i % 10 == 0:
                log(f"{i}/{len(todo)} files, {time.time() - t0:.0f}s, kept so far {len(sink.rows)} unflushed")
    sink.flush()
    summary = {**totals(out / "done"), "files_requested": len(files), "files_this_invocation": len(todo),
               "files_failed_this_invocation": len(status["errors"]), "wall_seconds": round(time.time() - t0, 1),
               "throttled": status["throttled"], "errors": status["errors"][:20],
               "filter": {"character": character, "ascension": ascension}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-files", type=int, default=None)
    ap.add_argument("--folder-prefix", default="", help="only files whose folder_path starts with this (e.g. Monthly_2020)")
    ap.add_argument("--exclude-runs", nargs="*", default=[],
                    help="earlier pull run dirs (or their out/): skip files already processed there (out/done/)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--filter-character", default="IRONCLAD")
    ap.add_argument("--filter-ascension", type=int, default=20)
    a = ap.parse_args()
    table = pq.read_table(a.files)
    if a.folder_prefix:
        import pyarrow.compute as pc
        table = table.filter(pc.starts_with(table["folder_path"], a.folder_prefix))
    table = exclude_done(table, a.exclude_runs)
    files = choose_files(table, a.n_files)
    s = run(files, Path(a.out), a.filter_character, a.filter_ascension, max(1, min(a.workers, 4)))
    print(json.dumps(s, indent=2))
    sys.exit(3 if s["throttled"] else 0)


if __name__ == "__main__":
    main()
