from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def read_parquet_records(
    path: str | Path, limit: int | None = None
) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)
    return table.to_pylist()


def read_parquet_row(path: str | Path, row_index: int = 0) -> dict[str, Any]:
    table = pq.read_table(path)
    if row_index < 0 or row_index >= table.num_rows:
        raise IndexError(
            f"row index {row_index} out of range (0..{table.num_rows - 1}) in {path}"
        )
    return table.slice(row_index, 1).to_pylist()[0]


def parquet_to_json(
    path: str | Path,
    row_index: int | None = None,
    limit: int | None = None,
    indent: int = 2,
) -> str:
    if row_index is not None:
        data = read_parquet_row(path, row_index)
    else:
        data = read_parquet_records(path, limit)
    return json.dumps(data, indent=indent)
