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

from sts_combat_rl.models.deep_sets import DeepSetsValue

from ..run import run_dir
from ..schemas import combat_v3
from .data import RowLoader, Rows, assign_targets, split_rows, training_rows


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


def predict(model: DeepSetsValue, loader: RowLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """(prediction, target) for the loader's rows, in loader order."""
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch.pop("weight", None)
            targets.append(batch.pop("target"))
            predictions.append(model(**batch))
    return torch.cat(predictions), torch.cat(targets)


def evaluate(model: DeepSetsValue, loader: RowLoader) -> tuple[float, float]:
    prediction, target = predict(model, loader)
    return ((prediction - target) ** 2).mean().item(), (
        prediction - target
    ).abs().mean().item()


def group_mse(rows: Rows, index: np.ndarray, prediction: torch.Tensor) -> dict[str, dict[str, float]]:
    """Per category/encounter: rows, model MSE, constant-mean MSE and teacher root_value MSE on `index`."""
    error = (prediction.numpy().astype(np.float64) - rows["target"][index]) ** 2
    target = rows["target"][index]
    teacher = (rows["root_value"][index].astype(np.float64) - target) ** 2
    keys = np.char.add(np.char.add(rows["category"][index].astype(str), "/"), rows["encounter"][index].astype(str))
    out = {}
    for key in np.unique(keys):
        mask = keys == key
        out[str(key)] = {"rows": int(mask.sum()), "mse": float(error[mask].mean()),
                         "baseline_mse": float(target[mask].var()), "teacher_mse": float(teacher[mask].mean())}
    return out


def pinned_split(rows: Rows, reference: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    """Bootstrap rows on the side the reference checkpoint put them: its train / validation episodes and
    run seeds (as row indices). Rows of other episodes are dropped."""
    split = []
    for ids, seeds in (("train_episode_ids", "train_run_seeds"), ("validation_episode_ids", "validation_run_seeds")):
        split.append(np.flatnonzero(np.isin(rows["episode_id"], reference[ids])
                                    & np.isin(rows["run_seed"], np.array(reference[seeds], dtype=np.uint64))))
    return split[0], split[1], len(rows) - len(split[0]) - len(split[1])


def load_corrections(sql, reference) -> Rows:
    """DAgger rows (apps/dagger) with target root_value (the teacher's estimate; never the actor's outcome).
    Uses the completed fights' parts even if the run itself didn't finish (each part is one whole fight).
    Every row must be a reference training episode / run seed."""
    rows = training_rows(sql, corrective=True)
    inside = (np.isin(rows["episode_id"], reference["train_episode_ids"])
              & np.isin(rows["run_seed"], np.array(reference["train_run_seeds"], dtype=np.uint64)))
    if outside := np.unique(rows["episode_id"][~inside]).tolist():
        raise ValueError(f"corrections outside the initial checkpoint's training episodes: {outside[:5]}")
    rows.columns["target"] = rows["root_value"].astype(np.float64)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(getattr(args, "threads", 1))  # intra-op CPU threads; 1 = deterministic default
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = training_rows(args.data, oracle=getattr(args, "oracle", False))
    assign_targets(rows, args.label, args.blend)
    corrections_sql = getattr(args, "corrections", None)
    initial = getattr(args, "initial_checkpoint", None)
    weight = getattr(args, "correction_weight", 0.5)
    split_mode = getattr(args, "split", "pinned")
    if split_mode not in ("pinned", "fresh"):
        raise ValueError(f"split {split_mode!r} is not 'pinned' or 'fresh'")
    pinned = bool(initial) and split_mode == "pinned"
    reference = None
    corrections = None
    if corrections_sql and not pinned:
        raise ValueError("corrections need an initial checkpoint with split = 'pinned'")
    if initial:
        reference = json.loads(Path(initial).with_suffix(".json").read_text())
        if reference["architecture"].get("width") != args.width:
            raise ValueError(f"width {args.width} != initial checkpoint architecture {reference['architecture']}")
    if pinned:
        # Pinned: same fights on each side as the initial checkpoint; validation stays bootstrap only.
        train, valid, dropped = pinned_split(rows, reference)
        kept = np.concatenate([train, valid])
        print(f"pinned split to {initial}: dropped {dropped:,} bootstrap rows outside it", flush=True)
    else:
        train, valid = split_rows(rows, args.validation_fraction, args.seed)
        kept = np.arange(len(rows))
    if not len(train) or not len(valid):
        raise ValueError("run split requires at least two run_seeds")
    train_ids = np.unique(rows["episode_id"][train]).tolist()
    valid_ids = np.unique(rows["episode_id"][valid]).tolist()
    train_runs = np.unique(rows["run_seed"][train]).tolist()
    valid_runs = np.unique(rows["run_seed"][valid]).tolist()
    row_counts = {
        column: dict(zip(*(a.tolist() for a in np.unique(rows[column][kept], return_counts=True))))
        for column in ("encounter", "category")
    }
    source_runs = sorted(set(rows["run_id"][kept]))
    bootstrap_train = train
    if corrections_sql:
        if not 0 <= weight <= 1:
            raise ValueError(f"correction_weight {weight} outside [0, 1]")
        corrections = load_corrections(corrections_sql, reference)
        # Per-row weights: the mean weighted loss over all training rows is
        # (1-w) * mean bootstrap loss + w * mean correction loss, whatever the row counts.
        total = len(train) + len(corrections)
        rows.columns["weight"][train] = (1 - weight) * total / len(train)
        corrections.columns["weight"][:] = weight * total / len(corrections)
        correction_ids = np.unique(corrections["episode_id"]).tolist()
        correction_runs = sorted(set(corrections["run_id"]))
        correction_count = len(corrections)
        offset = len(rows)
        rows = Rows.concat([rows, corrections])  # corrections are rows offset.. of the combined table
        corrections = np.arange(offset, len(rows))
        train = np.concatenate([train, corrections])
    train_loader = RowLoader(rows, train, args.batch_size, shuffle=True, weight=True,
                             generator=torch.Generator().manual_seed(args.seed))
    valid_loader = RowLoader(rows, valid, args.batch_size)
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
    lr_schedule = getattr(args, "lr_schedule", "constant")
    keep = getattr(args, "keep", "last")
    if lr_schedule not in ("constant", "cosine"):
        raise ValueError(f"lr_schedule {lr_schedule!r} is not 'constant' or 'cosine'")
    if keep not in ("last", "best"):
        raise ValueError(f"keep {keep!r} is not 'last' or 'best'")
    # cosine: per-step decay from lr to 0 over all epochs
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
                 if lr_schedule == "cosine" else None)
    targets = rows["target"]
    baseline = float(np.mean(targets[bootstrap_train]))
    component_loaders = {
        name: RowLoader(rows, part, args.batch_size)
        for name, part in (("bootstrap", bootstrap_train), ("correction", corrections)) if corrections is not None
    }
    # The teacher's own estimate as a predictor of the label: the bar the net should approach.
    teacher_mse = float(np.mean((rows["root_value"][valid].astype(np.float64) - targets[valid]) ** 2))
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
            "data_query": args.data,
            "source_runs": source_runs,
            "manifest": [json.loads((run_dir(r) / "run.json").read_text()) for r in source_runs],
            "project_git_revision": git_revision,
            "project_git_dirty": git_dirty,
            "split_mode": "pinned" if pinned else "fresh",
            "train_episode_ids": train_ids,
            "validation_episode_ids": valid_ids,
            "train_run_seeds": train_runs,
            "validation_run_seeds": valid_runs,
            "epoch": epoch,
            "metrics": metrics,
            "lr_schedule": lr_schedule,
            "keep": keep,
            "history": history,
        }
        if initial:
            checkpoint.update(
                initial_checkpoint=str(initial),
                initial_checkpoint_sha256=initial_sha,
                initialization="model weights from initial_checkpoint; fresh optimizer",
                split="pinned to initial_checkpoint train/validation episodes and run seeds" if pinned
                else "fresh episode_split(validation_fraction, seed) of the queried rows",
            )
        if corrections is not None:
            checkpoint.update(
                training_kind="corrective fine-tune (not a pure data ablation)",
                correction_weight=weight,
                correction_target="root_value (teacher_root_only)",
                correction_rows=correction_count,
                correction_episode_ids=correction_ids,
                correction_query=corrections_sql,
                correction_source=correction_runs,
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
        print(f"initial checkpoint: {initial} (weights only, fresh optimizer; split {split_mode})", flush=True)
    if corrections is not None:
        print(f"corrections: {correction_count:,} rows, {len(correction_ids)} fights, "
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
    history = []  # per-epoch metrics (incl. per-encounter validation MSE), also in the checkpoint json
    best_mse = float("inf")
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
            if scheduler is not None:
                scheduler.step()
            if batch_index == batch_count or batch_index % progress_every == 0:
                filled = round(24 * batch_index / batch_count)
                bar = "#" * filled + "." * (24 - filled)
                print(
                    f"  epoch {epoch:>3}/{args.epochs:<3} [{bar}] "
                    f"{batch_index:>5}/{batch_count:<5} loss={loss.item():.6f}",
                    flush=True,
                )
        train_mse, train_mae = evaluate(model, train_loader)
        valid_prediction, valid_target = predict(model, valid_loader)
        valid_mse = ((valid_prediction - valid_target) ** 2).mean().item()
        valid_mae = (valid_prediction - valid_target).abs().mean().item()
        baseline_mse = float(np.mean((targets[valid] - baseline) ** 2))
        metrics = {
            "train_mse": train_mse,
            "train_mae": train_mae,
            "validation_mse": valid_mse,
            "validation_mae": valid_mae,
            "baseline_validation_mse": baseline_mse,
            "teacher_root_value_validation_mse": teacher_mse,
        }
        if corrections is not None:
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
            f"teacher_mse={teacher_mse:.6f} "
            f"lr_end={optimizer.param_groups[0]['lr']:.2e}",
            flush=True,
        )
        groups = group_mse(rows, valid, valid_prediction)
        print(f"  {'validation by encounter':<32}{'rows':>8}{'mse':>10}{'baseline':>10}{'teacher':>10}", flush=True)
        for name, g in groups.items():
            print(f"  {name:<32}{g['rows']:>8}{g['mse']:>10.5f}{g['baseline_mse']:>10.5f}{g['teacher_mse']:>10.5f}",
                  flush=True)
        history.append({"epoch": epoch, "lr_end": optimizer.param_groups[0]["lr"], **metrics,
                        "validation_by_encounter": groups})
        metrics = {**metrics, "validation_by_encounter": groups}
        atomic_json(history, args.output.with_name("training_history.json"))
        if keep == "last" or valid_mse < best_mse:
            best_mse = min(best_mse, valid_mse)
            checkpoint = save_checkpoint(epoch, metrics)
            print(f"  saved {args.output} (epoch {epoch})", flush=True)
        else:
            print(f"  not saved: valid_mse {valid_mse:.6f} >= best {best_mse:.6f} (keep = best)", flush=True)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU single-threaded Deep Sets bootstrap value trainer"
    )
    parser.add_argument("data", help="SQL over sts_combat_rl.query, e.g. \"select * from combat_v3 where id = 'act1-a20'\"")
    parser.add_argument("--output", type=Path, default=Path("value_checkpoint.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--label",
        choices=["blend", "root", "terminal"],
        default="blend",
        help="training target (see data.assign_targets)",
    )
    parser.add_argument("--blend", type=float, default=0.5, help="weight on terminal_value for --label blend")
    parser.add_argument("--corrections", help="SQL selecting DAgger rows (target root_value)")
    parser.add_argument("--oracle", action="store_true", help="allow oracle (perfect-foresight) rows in the data")
    parser.add_argument("--initial-checkpoint", type=Path, help="start from these weights (fresh optimizer)")
    parser.add_argument("--split", choices=["pinned", "fresh"], default="pinned",
                        help="with --initial-checkpoint: pin to its train/validation episodes (dropping other rows), "
                             "or split the queried rows afresh (episode_split)")
    parser.add_argument("--correction-weight", type=float, default=0.5)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--keep", choices=["last", "best"], default="last",
                        help="checkpoint of the last epoch or of the best validation MSE")
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
