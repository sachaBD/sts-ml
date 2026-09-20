"""Evaluate a value checkpoint against its recorded held-out shard."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sts_combat_rl.models.deep_sets import DeepSetsValue

from .data import ValueDataset, collate_states, load_rows


def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_checkpoint_source(checkpoint: dict[str, Any], path: Path) -> None:
    if sha256(path) != checkpoint["source_sha256"]:
        raise ValueError("source shard SHA256 does not match checkpoint provenance")
    if (
        checkpoint.get("encoding_version") != 2
        or checkpoint.get("target_name") != "mcts_value"
    ):
        raise ValueError("checkpoint is not compatible with schema-v2 mcts_value")


def infer_boss_id(rows: list[dict[str, Any]]) -> int:
    identifiers = {
        token["monster_id"]
        for row in rows
        if row["decision_index"] == 0
        for token in row["monsters"]
    }
    if len(identifiers) != 1:
        raise ValueError(
            f"cannot infer a consistent boss ID from decision_index=0 rows: {sorted(identifiers)}"
        )
    return identifiers.pop()


def boss_slice_masks(
    rows: list[dict[str, Any]], boss_identifier: int
) -> dict[str, np.ndarray]:
    """Classify by boss HP; absent boss token means the split has occurred."""
    hp = np.array(
        [
            next(
                (
                    m["numeric"][1]
                    for m in row["monsters"]
                    if m["monster_id"] == boss_identifier
                ),
                np.nan,
            )
            for row in rows
        ]
    )
    return {
        "boss_above_60pct": hp > 0.6,
        "boss_near_split_40_60pct": (hp >= 0.4) & (hp <= 0.6),
        "boss_below_40pct": hp < 0.4,
        "post_split": np.isnan(hp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_source(checkpoint, args.data)
    all_rows = load_rows(args.data, args.limit)
    held_ids = set(checkpoint["validation_episode_ids"])
    rows = [row for row in all_rows if row["episode_id"] in held_ids]
    model = DeepSetsValue(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions: list[float] = []
    with torch.no_grad():
        loader = DataLoader(
            ValueDataset(rows),
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=collate_states,
        )
        for batch in loader:
            batch.pop("target")
            predictions.extend(model(**batch).tolist())
    targets = np.array([row["mcts_value"] for row in rows])
    predicted = np.array(predictions)
    train_ids = set(checkpoint["train_episode_ids"])
    train_mean = np.mean(
        [row["mcts_value"] for row in all_rows if row["episode_id"] in train_ids]
    )

    def report(name: str, mask: np.ndarray) -> None:
        y, p = targets[mask], predicted[mask]
        corr = (
            np.corrcoef(y, p)[0, 1]
            if len(y) > 1 and np.std(y) and np.std(p)
            else float("nan")
        )
        print(
            f"{name}: n={len(y)} mse={np.mean((p - y) ** 2):.6f} mae={np.mean(abs(p - y)):.6f} pearson={corr:.6f} baseline_mse={np.mean((y - train_mean) ** 2):.6f}"
        )

    report("held_out", np.ones(len(rows), dtype=bool))
    decision = np.array([row["decision_index"] for row in rows])
    for name, mask in (
        ("phase_early", decision < 5),
        ("phase_middle", (decision >= 5) & (decision < 15)),
        ("phase_late", decision >= 15),
    ):
        if mask.any():
            report(name, mask)
    for name, mask in boss_slice_masks(rows, infer_boss_id(all_rows)).items():
        if mask.any():
            report(name, mask)


if __name__ == "__main__":
    main()
