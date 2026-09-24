"""The one way to get combat_v3 rows for training: load_rows, refusing oracle data.

Oracle rows (runs/README.md `oracle`) were played with perfect RNG foresight: an upper bound, not fair
play, so their outcomes and root values are too optimistic to train on. Use load_rows directly for
analysis / evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.dataset as ds


def get_training_samples(
    path: str | Path,
    limit: int | None = None,
    categories: list[str] | None = None,
    encounters: list[str] | None = None,
    corrective: bool = False,
) -> list[dict[str, Any]]:
    """load_rows (same arguments), but raises if any part holds oracle = true rows."""
    from ..training.data import load_rows  # lazy: pulls in torch

    # Per part: a dataset-level read uses the first part's schema and can drop the column entirely.
    for fragment in ds.dataset(path, format="parquet", exclude_invalid_files=True).get_fragments():
        if "oracle" in fragment.physical_schema.names:
            if pc.any(fragment.to_table(columns=["oracle"])["oracle"]).as_py():
                raise ValueError(f"{fragment.path}: oracle rows (perfect-foresight teacher) are not training data")
    return load_rows(path, limit, categories, encounters, corrective)
