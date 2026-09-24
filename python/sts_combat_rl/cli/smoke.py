from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ..data.dataset import generate_dataset
from ..data.reader import read_parquet_row
from ..models.deep_sets import DeepSetsValue
from ..models.encoding import value_tensors


def run_smoke(parquet_path: Path | None = None) -> None:
    root = Path(__file__).resolve().parents[3]
    if parquet_path is None:
        default_probe = (
            root / "data" / "mcts-slime-v2-probe-4x100" / "mcts_slime_v2.parquet"
        )
        if default_probe.exists():
            parquet_path = default_probe
        else:
            smoke_dir = root / "data" / "mcts-slime-smoke"
            shard, _, _ = generate_dataset(
                output_dir=smoke_dir,
                seed_start=1,
                seed_count=1,
                simulations=10,
                generator=str(root / "build" / "main" / "generate_mcts_records"),
            )
            parquet_path = shard

    state = read_parquet_row(parquet_path, 0)
    print(json.dumps(state, indent=2))
    tensors = value_tensors(state)
    torch.manual_seed(0)
    model = DeepSetsValue()
    print(model)
    print({name: tuple(value.shape) for name, value in tensors.items()})
    print("value:", model(**tensors).item())


def main(argv: list[str] | None = None) -> None:
    parquet_path = (
        Path(argv[0])
        if argv and len(argv) > 0
        else (Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    )
    run_smoke(parquet_path)


if __name__ == "__main__":
    main()
