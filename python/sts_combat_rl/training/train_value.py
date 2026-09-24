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
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from sts_combat_rl.models.deep_sets import DeepSetsValue

from ..schemas import combat_v3
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
            batch.pop("weight", None)
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


def weighted_collate(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """collate_states plus each row's loss weight (1 unless corrective training set one)."""
    batch = collate_states(rows)
    batch["weight"] = torch.tensor([r.get("weight", 1.0) for r in rows], dtype=torch.float32)
    return batch


def pinned_split(rows: list[dict[str, Any]], reference: dict[str, Any]):
    """Bootstrap rows on the side the reference checkpoint put them: its train / validation episodes and
    run seeds. Rows of other episodes are dropped."""
    split = {}
    for side, ids, seeds in (("train", "train_episode_ids", "train_run_seeds"),
                             ("validation", "validation_episode_ids", "validation_run_seeds")):
        ids, seeds = set(reference[ids]), set(reference[seeds])
        split[side] = [r for r in rows if r["episode_id"] in ids and r["run_seed"] in seeds]
    kept = len(split["train"]) + len(split["validation"])
    return split["train"], split["validation"], len(rows) - kept


def load_corrections(paths, reference, args) -> list[dict[str, Any]]:
    """DAgger rows (apps/dagger) with target root_value (the teacher's estimate; never the actor's outcome).
    Uses the completed fights' parts even if the run itself didn't finish (each part is one whole fight).
    Every row must be a reference training episode / run seed."""
    rows = []
    train_ids, train_seeds = set(reference["train_episode_ids"]), set(reference["train_run_seeds"])
    for path in paths:
        part = load_rows(path, args.limit, args.categories, args.encounters, corrective=True)
        if outside := sorted({r["episode_id"] for r in part
                              if r["episode_id"] not in train_ids or r["run_seed"] not in train_seeds}):
            raise ValueError(f"{path}: corrections outside the initial checkpoint's training episodes: {outside[:5]}")
        rows.extend(part)
    if not rows:
        raise ValueError("corrections have no rows")
    for r in rows:
        r["target"] = r["root_value"]
        r["source"] = "correction"
    return rows


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
    corrections_paths = getattr(args, "corrections", None) or []
    initial = getattr(args, "initial_checkpoint", None)
    weight = getattr(args, "correction_weight", 0.5)
    reference = None
    corrections: list[dict[str, Any]] = []
    if initial:
        reference = json.loads(Path(initial).with_suffix(".json").read_text())
        if reference["architecture"].get("width") != args.width:
            raise ValueError(f"width {args.width} != initial checkpoint architecture {reference['architecture']}")
        # Pinned: same fights on each side as the initial checkpoint; validation stays bootstrap only.
        train, valid, dropped = pinned_split(rows, reference)
        rows = train + valid
        print(f"pinned split to {initial}: dropped {dropped:,} bootstrap rows outside it", flush=True)
    elif corrections_paths:
        raise ValueError("corrections need an initial checkpoint (its split is pinned)")
    else:
        train, valid, _, _ = episode_split(rows, args.validation_fraction, args.seed)
    if not train or not valid:
        raise ValueError("run split requires at least two run_seeds")
    train_ids = sorted({r["episode_id"] for r in train})
    valid_ids = sorted({r["episode_id"] for r in valid})
    train_runs = sorted({row["run_seed"] for row in train})
    valid_runs = sorted({row["run_seed"] for row in valid})
    row_counts = {
        column: dict(sorted(Counter(row[column] for row in rows).items()))
        for column in ("encounter", "category")
    }
    bootstrap_train = train
    if corrections_paths:
        if not 0 <= weight <= 1:
            raise ValueError(f"correction_weight {weight} outside [0, 1]")
        corrections = load_corrections(corrections_paths, reference, args)
        # Per-row weights: the mean weighted loss over all training rows is
        # (1-w) * mean bootstrap loss + w * mean correction loss, whatever the row counts.
        total = len(train) + len(corrections)
        for r in train:
            r["weight"] = (1 - weight) * total / len(train)
        for r in corrections:
            r["weight"] = weight * total / len(corrections)
        train = train + corrections
    train_loader = DataLoader(
        ValueDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=weighted_collate,
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
    initial_sha = None
    if initial:
        state = torch.load(initial, map_location="cpu", weights_only=False)
        if state["architecture"] != model.config:
            raise ValueError(f"initial architecture {state['architecture']} != {model.config}")
        model.load_state_dict(state["model_state"])  # weights only; the optimizer starts fresh
        initial_sha = hashlib.sha256(Path(initial).read_bytes()).hexdigest()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    baseline = float(np.mean([row["target"] for row in bootstrap_train]))
    component_loaders = {
        name: DataLoader(ValueDataset(part), batch_size=args.batch_size, shuffle=False, num_workers=0,
                         collate_fn=collate_states)
        for name, part in (("bootstrap", bootstrap_train), ("correction", corrections)) if corrections
    }
    # The teacher's own estimate as a predictor of the label: the bar the net should approach.
    teacher_mse = float(
        np.mean([(row["root_value"] - row["target"]) ** 2 for row in valid])
    )
    provenance = [_provenance(Path(path)) for path in args.data]
    correction_provenance = [_provenance(Path(path)) for path in corrections_paths]
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
            "data_schema": combat_v3.NAME,
            "row_counts": row_counts,
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
            "train_run_seeds": train_runs,
            "validation_run_seeds": valid_runs,
            "epoch": epoch,
            "metrics": metrics,
        }
        if initial:
            checkpoint.update(
                initial_checkpoint=str(initial),
                initial_checkpoint_sha256=initial_sha,
                initialization="model weights from initial_checkpoint; fresh optimizer",
                split="pinned to initial_checkpoint train/validation episodes and run seeds",
            )
        if corrections:
            checkpoint.update(
                training_kind="corrective fine-tune (not a pure data ablation)",
                correction_weight=weight,
                correction_target="root_value (teacher_root_only)",
                correction_rows=len(corrections),
                correction_episode_ids=sorted({r["episode_id"] for r in corrections}),
                correction_source=[str(path) for path in corrections_paths],
                correction_sha256=[p[0] for p in correction_provenance],
                correction_manifest_path=[str(p[1]) for p in correction_provenance],
                validation="bootstrap only, original targets, unweighted",
            )
        atomic_save(checkpoint, args.output)
        atomic_json(
            {key: checkpoint[key] for key in checkpoint if key != "model_state"},
            args.output.with_suffix(".json"),
        )
        return checkpoint

    print("\nDataset", flush=True)
    print("-------", flush=True)
    print(f"rows: train={len(train):,}  valid={len(valid):,}", flush=True)
    print(f"runs: train={len(train_runs):,}  valid={len(valid_runs):,}", flush=True)
    print(f"fights: train={len(train_ids):,}  valid={len(valid_ids):,}", flush=True)
    for column, counts in row_counts.items():
        print(f"{column}: {counts}", flush=True)
    print(f"target: label={args.label}  blend={args.blend}", flush=True)
    if initial:
        print(f"initial checkpoint: {initial} (weights only, fresh optimizer; split pinned)", flush=True)
    if corrections:
        print(f"corrections: {len(corrections):,} rows, {len({r['episode_id'] for r in corrections})} fights, "
              f"target root_value, weight {weight}", flush=True)

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
            weights = batch.pop("weight")
            optimizer.zero_grad()
            loss = (weights * (model(**batch) - target) ** 2).mean()
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
        if corrections:
            components = {name: evaluate(model, loader)[0] for name, loader in component_loaders.items()}
            metrics.update(
                train_bootstrap_mse=components["bootstrap"],
                train_correction_mse=components["correction"],
                train_objective=(1 - weight) * components["bootstrap"] + weight * components["correction"],
            )
            print(f"  train components: bootstrap_mse={components['bootstrap']:.6f} "
                  f"correction_mse={components['correction']:.6f} (weight {weight})", flush=True)
        elapsed = time.monotonic() - started
        print(
            "  summary "
            f"epoch={epoch}/{args.epochs} "
            f"time={elapsed:.1f}s "
            f"train_mse={train_mse:.6f} "
            f"train_mae={train_mae:.6f} "
            f"valid_mse={valid_mse:.6f} "
            f"valid_mae={valid_mae:.6f} "
            f"baseline_mse={baseline_mse:.6f} "
            f"teacher_mse={teacher_mse:.6f}",
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
    parser.add_argument("--categories", nargs="+", help="keep only these categories, e.g. boss elite")
    parser.add_argument("--encounters", nargs="+", help="keep only these encounters, e.g. slime_boss")
    parser.add_argument(
        "--label",
        choices=["blend", "root", "terminal"],
        default="blend",
        help="training target (see data.assign_targets)",
    )
    parser.add_argument("--blend", type=float, default=0.5, help="weight on terminal_value for --label blend")
    parser.add_argument("--corrections", type=Path, nargs="+", help="DAgger run out/ dirs (target root_value)")
    parser.add_argument("--initial-checkpoint", type=Path, help="start from these weights; pins the split")
    parser.add_argument("--correction-weight", type=float, default=0.5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
