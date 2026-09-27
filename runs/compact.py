#!/usr/bin/env python3
"""Merge small .parquet files into ~128MB ones, in <dir> and every subdirectory.

Usage: .venv/bin/python runs/compact.py <dir> [target_mb=128]

Each directory is compacted on its own (files never move between directories).
Crash safe: new files are written as *.tmp, then a journal is committed, and only
then are the originals deleted. Just rerun after a crash.
"""
import json, os, sys, time
import pyarrow.parquet as pq


def fsync(path):
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def finish(d, journal):
    j = json.load(open(journal))
    for tmp, final in j["outputs"]:
        if os.path.exists(tmp):
            os.rename(tmp, final)
    fsync(d)
    for f in j["inputs"]:
        if os.path.exists(f):
            os.remove(f)
    os.remove(journal)
    fsync(d)


def compact(d, target):
    journal = os.path.join(d, "_compact.journal")

    # Recover from a crash: finish a committed job, drop uncommitted leftovers.
    if os.path.exists(journal):
        finish(d, journal)
    for f in os.listdir(d):
        if f.endswith(".tmp"):
            os.remove(os.path.join(d, f))

    inputs = sorted(
        p for f in os.listdir(d)
        if f.endswith(".parquet") and os.path.getsize(p := os.path.join(d, f)) < target
    )
    if len(inputs) < 2:
        return

    # Write new files as .tmp, starting a new one every ~target bytes.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outputs, writer = [], None
    for f in inputs:
        table = pq.read_table(f)
        if writer is None:
            final = os.path.join(d, f"compact-{stamp}-{len(outputs):04d}.parquet")
            outputs.append((final + ".tmp", final))
            writer = pq.ParquetWriter(final + ".tmp", table.schema, compression="zstd")
        writer.write_table(table)  # raises on schema mismatch -> nothing deleted
        if os.path.getsize(outputs[-1][0]) >= target:
            writer.close()
            writer = None
    if writer is not None:
        writer.close()
    for tmp, _ in outputs:
        fsync(tmp)

    n_in = sum(pq.ParquetFile(f).metadata.num_rows for f in inputs)
    n_out = sum(pq.ParquetFile(t).metadata.num_rows for t, _ in outputs)
    assert n_in == n_out, f"{d}: row mismatch {n_in} != {n_out}, originals untouched"

    # Commit, then swap.
    with open(journal + ".tmp", "w") as fh:
        json.dump({"outputs": outputs, "inputs": inputs}, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.rename(journal + ".tmp", journal)
    fsync(d)
    finish(d, journal)
    print(f"{d}: {len(inputs)} -> {len(outputs)} files, {n_out} rows")


target = int(sys.argv[2] if len(sys.argv) > 2 else 128) * 1024 * 1024
for d, _, _ in os.walk(sys.argv[1]):
    compact(d, target)
