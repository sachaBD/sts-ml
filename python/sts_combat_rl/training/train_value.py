"""Deterministic, single-threaded CPU trainer for bootstrap values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sts_combat_rl.models.deep_sets import DeepSetsValue

from ..data.entry_roots import deck_signature_split
from .data import ValueDataset, assign_targets, collate_states, episode_split, load_rows


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


def _run_json(source: Path) -> Path | None:
    """The run.json of the run this data came from: runs/schema=/date=/id=/out/... (runs/README.md)."""
    for parent in [source, *source.parents]:
        if parent.name.startswith("id=") and (parent / "run.json").exists():
            return parent / "run.json"
    return None


def _provenance(source: Path) -> tuple[str, Path | None, Any]:
    """(sha256, run.json path, run.json) for one Parquet shard or a directory of parts."""
    record = _run_json(source.resolve())
    metadata = json.loads(record.read_text()) if record else {}
    if source.is_dir():
        digest = hashlib.sha256()
        for f in sorted(source.rglob("*.parquet")):
            digest.update(str(f.relative_to(source)).encode())
            digest.update(hashlib.sha256(f.read_bytes()).digest())
        return digest.hexdigest(), record, metadata
    return hashlib.sha256(source.read_bytes()).hexdigest(), record, metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [
        row
        for path in args.data
        for row in load_rows(path, args.limit, args.categories, args.encounters)
    ]
    if not rows:
        raise ValueError("no rows match the data paths and category/encounter filters")
    assign_targets(rows, args.label, args.blend)
    # Entry-root shards must split by canonical deck signature so decisions and
    # combat replicates from the same natural deck never leak across folds.
    train_signature_ids, valid_signature_ids = [], []
    deck_signatures = (
        {row.get("deck_signature") for row in rows if row.get("deck_signature") is not None}
        if rows and "deck_signature" in rows[0]
        else set()
    )
    if len(deck_signatures) >= 2:
        train, valid, train_signature_ids, valid_signature_ids = deck_signature_split(
            rows, args.validation_fraction, args.seed
        )
        train_ids = sorted({row["episode_id"] for row in train})
        valid_ids = sorted({row["episode_id"] for row in valid})
    else:
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
    baseline = float(np.mean([row["target"] for row in train]))
    # The teacher's own estimate as a predictor of the label: the bar the net should approach.
    teacher_mse = (
        float(np.mean([(row["root_value"] - row["target"]) ** 2 for row in valid]))
        if "root_value" in valid[0]
        else float("nan")
    )
    provenance = [_provenance(Path(path)) for path in args.data]
    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    git_dirty = (
        subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0
    )

    def save_checkpoint(epoch: int, metrics: dict[str, float]) -> dict[str, Any]:
        checkpoint = {
            "model_state": model.state_dict(),
            "architecture": model.config,
            "optimizer_config": {
                "name": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "training_config": vars(args),
            "encoding_version": 3,
            "target_name": args.label,
            "target_blend": args.blend,
            "source_shard": [str(path) for path in args.data],
            "source_sha256": [p[0] for p in provenance],
            "manifest_path": [str(p[1]) for p in provenance],
            "manifest": [p[2] for p in provenance],
            "project_git_revision": git_revision,
            "project_git_dirty": git_dirty,
            "train_episode_ids": train_ids,
            "validation_episode_ids": valid_ids,
            "train_deck_signatures": train_signature_ids
            if rows and "deck_signature" in rows[0]
            else [],
            "validation_deck_signatures": valid_signature_ids
            if rows and "deck_signature" in rows[0]
            else [],
            "epoch": epoch,
            "metrics": metrics,
        }
        atomic_save(checkpoint, args.output)
        atomic_json(
            {key: checkpoint[key] for key in checkpoint if key != "model_state"},
            args.output.with_suffix(".json"),
        )
        return checkpoint

    print("\nDataset", flush=True)
    print("-------", flush=True)
    print(f"rows: train={len(train):,}  valid={len(valid):,}", flush=True)
    if train_signature_ids or valid_signature_ids:
        print(
            f"decks: train={len(train_signature_ids):,}  valid={len(valid_signature_ids):,}",
            flush=True,
        )
    else:
        print(
            f"episodes: train={len(train_ids):,}  valid={len(valid_ids):,}",
            flush=True,
        )
    print(f"target: label={args.label}  blend={args.blend}", flush=True)

    print("\nTraining", flush=True)
    print("--------", flush=True)
    print("Each epoch prints batch progress, then validation metrics.", flush=True)
    print(
        "epoch  time    train_mse  train_mae  valid_mse  valid_mae  baseline   teacher",
        flush=True,
    )
    print(
        "-----  ------  ---------  ---------  ---------  ---------  ---------  ---------",
        flush=True,
    )
    metrics = {}
    checkpoint = {}
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        model.train()
        batch_count = len(train_loader)
        progress_every = max(1, batch_count // 20)
        for batch_index, batch in enumerate(train_loader, start=1):
            target = batch.pop("target")
            optimizer.zero_grad()
            loss = ((model(**batch) - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            if batch_index == batch_count or batch_index % progress_every == 0:
                filled = round(24 * batch_index / batch_count)
                bar = "#" * filled + "." * (24 - filled)
                print(
                    f"  epoch {epoch:>3}/{args.epochs:<3} [{bar}] "
                    f"{batch_index:>5}/{batch_count:<5} loss={loss.item():.6f}",
                    flush=True,
                )
        train_mse, train_mae = evaluate(model, train_loader)
        valid_mse, valid_mae = evaluate(model, valid_loader)
        baseline_mse = float(
            np.mean([(row["target"] - baseline) ** 2 for row in valid])
        )
        metrics = {
            "train_mse": train_mse,
            "train_mae": train_mae,
            "validation_mse": valid_mse,
            "validation_mae": valid_mae,
            "baseline_validation_mse": baseline_mse,
            "teacher_root_value_validation_mse": teacher_mse,
        }
        elapsed = time.monotonic() - started
        teacher = f"{teacher_mse:9.6f}" if np.isfinite(teacher_mse) else "      n/a"
        print(
            "  summary "
            f"epoch={epoch}/{args.epochs} "
            f"time={elapsed:.1f}s "
            f"train_mse={train_mse:.6f} "
            f"train_mae={train_mae:.6f} "
            f"valid_mse={valid_mse:.6f} "
            f"valid_mae={valid_mae:.6f} "
            f"baseline_mse={baseline_mse:.6f} "
            f"teacher_mse={teacher.strip()}",
            flush=True,
        )
        checkpoint = save_checkpoint(epoch, metrics)
        print(f"  saved {args.output} (epoch {epoch})", flush=True)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU single-threaded Deep Sets bootstrap value trainer"
    )
    parser.add_argument("data", type=Path, nargs="+", help="Parquet shards or run out/ directories")
    parser.add_argument("--output", type=Path, default=Path("value_checkpoint.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--categories", nargs="+", help="keep only these combat_v3 categories, e.g. boss elite")
    parser.add_argument("--encounters", nargs="+", help="keep only these encounters, e.g. slime_boss")
    parser.add_argument(
        "--label",
        choices=["blend", "root", "terminal", "mcts"],
        default="blend",
        help="training target (see data.assign_targets); mcts = legacy mcts_value column",
    )
    parser.add_argument("--blend", type=float, default=0.5, help="weight on terminal_value for --label blend")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
