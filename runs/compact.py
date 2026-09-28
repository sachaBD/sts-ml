#!/usr/bin/env python3
"""Merge small .parquet files into ~128MB ones, in <dir> and every subdirectory.

Usage: .venv/bin/python runs/compact.py <dir> [target_mb=128]

Each directory is compacted on its own (files never move between directories).
Crash safe: new files are written as *.tmp, then a journal is committed, and only
then are the originals deleted. Just rerun after a crash.
"""
import json, os, sys, time
import pyarrow as pa
import pyarrow.parquet as pq

# Rows per output row group. Each input file (often one fight) used to become its own row group, so compacted
# files had thousands of ~100-row groups and ~20MB footers that duckdb parses on every query.
ROW_GROUP_ROWS = 128 * 1024


def tiny_row_groups(path):
    """Many row groups averaging well under ROW_GROUP_ROWS (e.g. written by the old compact.py)."""
    m = pq.ParquetFile(path).metadata
    return m.num_row_groups > 1 and m.num_rows / m.num_row_groups < ROW_GROUP_ROWS / 4


def needs_compaction(path, target):
    return os.path.getsize(path) < target or tiny_row_groups(path)


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
        if f.endswith(".parquet") and needs_compaction(p := os.path.join(d, f), target)
    )
    if not inputs or (len(inputs) == 1 and not tiny_row_groups(inputs[0])):
        return

    # Write new files as .tmp, buffering rows into ROW_GROUP_ROWS row groups, starting a new file every ~target bytes.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outputs, writer, schema, buffer, buffered = [], None, None, [], 0

    def flush():  # writes one row group of up to ROW_GROUP_ROWS rows; the rest stays buffered
        nonlocal writer, buffer, buffered
        if not buffer:
            return
        if writer is None:
            final = os.path.join(d, f"compact-{stamp}-{len(outputs):04d}.parquet")
            outputs.append((final + ".tmp", final))
            writer = pq.ParquetWriter(final + ".tmp", schema, compression="zstd")
        table = pa.Table.from_batches(buffer, schema)
        writer.write_table(table.slice(0, ROW_GROUP_ROWS), row_group_size=ROW_GROUP_ROWS)
        rest = table.slice(ROW_GROUP_ROWS)
        buffer, buffered = rest.to_batches(), rest.num_rows
        if os.path.getsize(outputs[-1][0]) >= target:
            writer.close()
            writer = None

    for f in inputs:
        pf = pq.ParquetFile(f)
        if schema is None:
            schema = pf.schema_arrow
        elif not pf.schema_arrow.equals(schema):  # -> nothing deleted
            raise ValueError(f"{f}: schema differs from {inputs[0]}")
        for batch in pf.iter_batches(batch_size=16384):
            buffer.append(batch)
            buffered += batch.num_rows
            if buffered >= ROW_GROUP_ROWS:
                flush()
    while buffer:
        flush()
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


if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
    sys.exit(__doc__)
target = int(sys.argv[2] if len(sys.argv) > 2 else 128) * 1024 * 1024
for d, _, _ in os.walk(sys.argv[1]):
    compact(d, target)
