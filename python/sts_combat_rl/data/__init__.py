"""Parquet readers and lazily imported dataset generation helpers."""

from .reader import parquet_to_json, read_parquet_records, read_parquet_row

__all__ = [
    "SCHEMA",
    "generate_dataset",
    "parquet_to_json",
    "read_parquet_records",
    "read_parquet_row",
    "validate_records",
]


def __getattr__(name: str):
    if name in {"SCHEMA", "generate_dataset", "validate_records"}:
        from . import dataset

        return getattr(dataset, name)
    raise AttributeError(name)
