"""Deterministic, single-threaded CPU trainer for bootstrap values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tomllib
import torch
from torch.utils.data import DataLoader

from sts_combat_rl.models.deep_sets import DeepSetsValue

from .data import ValueDataset, collate_states, episode_split, load_rows


def atomic_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.name, suffix=".tmp"
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    )
    os.replace(temporary, path)


def evaluate(model: DeepSetsValue, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            targets.append(batch.pop("target"))
            predictions.append(model(**batch))
    target, prediction = torch.cat(targets), torch.cat(predictions)
    return ((prediction - target) ** 2).mean().item(), (
        prediction - target
    ).abs().mean().item()


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_rows(args.data, args.limit)
    train, valid, train_ids, valid_ids = episode_split(
        rows, args.validation_fraction, args.seed
    )
    if not train or not valid:
        raise ValueError("episode split requires at least two episodes")
    train_loader = DataLoader(
        ValueDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_states,
        generator=torch.Generator().manual_seed(args.seed),
    )
    valid_loader = DataLoader(
        ValueDataset(valid),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_states,
    )
    model = DeepSetsValue(width=args.width)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    baseline = float(np.mean([row["mcts_value"] for row in train]))
    metrics = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            target = batch.pop("target")
            optimizer.zero_grad()
            loss = ((model(**batch) - target) ** 2).mean()
            loss.backward()
            optimizer.step()
        train_mse, train_mae = evaluate(model, train_loader)
        valid_mse, valid_mae = evaluate(model, valid_loader)
        baseline_mse = float(
            np.mean([(row["mcts_value"] - baseline) ** 2 for row in valid])
        )
        metrics = {
            "train_mse": train_mse,
            "train_mae": train_mae,
            "validation_mse": valid_mse,
            "validation_mae": valid_mae,
            "baseline_validation_mse": baseline_mse,
        }
        print(
            f"epoch={epoch} train_mse={train_mse:.6f} train_mae={train_mae:.6f} validation_mse={valid_mse:.6f} validation_mae={valid_mae:.6f} baseline_mse={baseline_mse:.6f}",
            flush=True,
        )
    source = Path(args.data)
    manifest_path = source.parent / "manifest.toml"
    manifest = (
        tomllib.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    checkpoint = {
        "model_state": model.state_dict(),
        "architecture": model.config,
        "optimizer_config": {
            "name": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "training_config": vars(args),
        "encoding_version": 2,
        "target_name": "mcts_value",
        "source_shard": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "project_git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "project_git_dirty": subprocess.run(
            ["git", "diff", "--quiet"], check=False
        ).returncode
        != 0,
        "train_episode_ids": train_ids,
        "validation_episode_ids": valid_ids,
        "epoch": args.epochs,
        "metrics": metrics,
    }
    atomic_save(checkpoint, args.output)
    atomic_json(
        {key: checkpoint[key] for key in checkpoint if key != "model_state"},
        args.output.with_suffix(".json"),
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU single-threaded Deep Sets bootstrap value trainer"
    )
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("value_checkpoint.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
