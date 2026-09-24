#!/usr/bin/env python3
"""Config-driven value-network training app.

This is a thin TOML wrapper around sts_combat_rl.training.train_value for use
with the repository run launcher (apps/common/launch.sh).  It resolves configured combat-data inputs to
Parquet locations, writes a checkpoint into --out, optionally exports native C++
weights, and emits summary.json for run.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from apps.common.app import main, value_run, write_json
from sts_combat_rl.run import REPO, run_dir
from sts_combat_rl.training.export_value_weights import export as export_weights
from sts_combat_rl.training.train_value import run as train_value

def configured_inputs(config: dict[str, Any]) -> list[str]:
    """Bootstrap data: [data].bootstrap (corrective configs), else [run].input(s) / [data].paths."""
    run = config.get("run", {})
    if "bootstrap" in config.get("data", {}):
        if "input" in run or "inputs" in run or "paths" in config["data"]:
            raise ValueError("set [data].bootstrap or [run].input(s)/[data].paths, not both")
        run = {"inputs": config["data"]["bootstrap"]}
    value = run.get("inputs", run.get("input"))
    if value is None:
        value = config.get("data", {}).get("paths")
    if value is None:
        raise ValueError("config must set [run].input, [run].inputs, or [data].paths")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    raise TypeError("inputs must be a string or list of strings")


def resolve_data_path(value: str) -> Path:
    """Resolve a run_id/path to the data path accepted by train_value.

    run_id inputs resolve to that run's out/ directory.  Existing run directories
    also resolve to out/.  Existing paths to out/ directories, parquet files, or
    arbitrary parquet directories are passed through.
    """
    if value.count("/") == 2:
        try:
            return run_dir(value) / "out"
        except FileNotFoundError:
            pass
    path = Path(value)
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        raise FileNotFoundError(f"input {value!r} is not an existing path or run_id")
    if path.is_dir() and (path / "run.json").exists():
        return path / "out"
    return path


def optional_int(table: dict[str, Any], key: str) -> int | None:
    value = table.get(key)
    return None if value is None else int(value)


def optional_names(table: dict[str, Any], key: str) -> list[str] | None:
    """[data].categories / [data].encounters: a name or list of names; absent = no filter."""
    value = table.get(key)
    return [value] if isinstance(value, str) else value


def initial_checkpoint(value: str | None) -> Path | None:
    """[train].initial_checkpoint: a value_net_v1 run_id (-> its finished checkpoint) or a checkpoint path."""
    if value is None:
        return None
    path = resolve_data_path(value)
    return value_run(value).checkpoint if path.is_dir() else path


def make_train_args(config: dict[str, Any], out: Path) -> SimpleNamespace:
    train = config.get("train", {})
    data = config.get("data", {})
    unknown = sorted(set(data) - {"bootstrap", "corrections", "paths", "categories", "encounters"})
    if unknown:
        raise ValueError(f"unknown [data] keys: {unknown}")
    return SimpleNamespace(
        corrections=[resolve_data_path(x) for x in data.get("corrections", [])],
        initial_checkpoint=initial_checkpoint(train.get("initial_checkpoint")),
        correction_weight=float(train.get("correction_weight", 0.5)),
        categories=optional_names(data, "categories"),
        encounters=optional_names(data, "encounters"),
        data=[resolve_data_path(x) for x in configured_inputs(config)],
        output=out / str(train.get("checkpoint", "value_checkpoint.pt")),
        epochs=int(train.get("epochs", 20)),
        batch_size=int(train.get("batch_size", 128)),
        lr=float(train.get("lr", 0.001)),
        weight_decay=float(train.get("weight_decay", 0.0001)),
        width=int(train.get("width", 64)),
        seed=int(train.get("seed", 0)),
        validation_fraction=float(train.get("validation_fraction", 0.2)),
        limit=optional_int(train, "limit"),
        label=str(train.get("label", "blend")),  # blend, root or terminal
        blend=float(train.get("blend", 0.5)),
    )


def write_summary(out: Path, checkpoint_path: Path, checkpoint_json: Path, data_paths: list[Path], exported: Path | None) -> None:
    metadata = json.loads(checkpoint_json.read_text())
    summary = {
        "checkpoint": checkpoint_path.name,
        "checkpoint_json": checkpoint_json.name,
        "weights": exported.name if exported else None,
        "data": [str(p) for p in data_paths],
        "data_schema": metadata["data_schema"],
        "row_counts": metadata["row_counts"],
        "categories": metadata.get("training_config", {}).get("categories"),
        "encounters": metadata.get("training_config", {}).get("encounters"),
        "target_name": metadata.get("target_name"),
        "target_blend": metadata.get("target_blend"),
        "epoch": metadata.get("epoch"),
        "metrics": metadata.get("metrics"),
        "train_episodes": len(metadata.get("train_episode_ids", [])),
        "validation_episodes": len(metadata.get("validation_episode_ids", [])),
        "train_runs": len(metadata["train_run_seeds"]),
        "validation_runs": len(metadata["validation_run_seeds"]),
    }
    for key in ("initial_checkpoint", "initial_checkpoint_sha256", "initialization", "split", "training_kind",
                "correction_weight", "correction_target", "correction_rows", "correction_source", "correction_sha256"):
        if key in metadata:
            summary[key] = metadata[key]
    if "correction_episode_ids" in metadata:
        summary["correction_episodes"] = len(metadata["correction_episode_ids"])
    write_json(out / "summary.json", summary)


def inputs(config: dict[str, Any]) -> list[str]:
    """Every run this trains from: the initial checkpoint, the data inputs, the corrections."""
    initial = config.get("train", {}).get("initial_checkpoint")
    return [*([initial] if initial else []), *configured_inputs(config), *config.get("data", {}).get("corrections", [])]


def train(config: dict[str, Any], config_path: Path, out: Path) -> int:
    train_args = make_train_args(config, out)

    print("\nValue training", flush=True)
    print("==============", flush=True)
    print(f"config:     {config_path}", flush=True)
    print(f"output:     {out}", flush=True)
    print(f"checkpoint: {train_args.output}", flush=True)
    print("\nInputs", flush=True)
    print("------", flush=True)
    for path in train_args.data:
        print(f"- {path}", flush=True)
    for path in train_args.corrections:
        print(f"- {path} (correction)", flush=True)
    if train_args.initial_checkpoint:
        print(f"- {train_args.initial_checkpoint} (initial checkpoint)", flush=True)
    print("\nConfig", flush=True)
    print("------", flush=True)
    print(f"epochs:              {train_args.epochs}", flush=True)
    print(f"batch_size:          {train_args.batch_size}", flush=True)
    print(f"lr:                  {train_args.lr}", flush=True)
    print(f"weight_decay:        {train_args.weight_decay}", flush=True)
    print(f"width:               {train_args.width}", flush=True)
    print(f"seed:                {train_args.seed}", flush=True)
    print(f"validation_fraction: {train_args.validation_fraction}", flush=True)
    print(f"label:               {train_args.label}", flush=True)
    print(f"blend:               {train_args.blend}", flush=True)
    print(f"categories:          {train_args.categories or 'all'}", flush=True)
    print(f"encounters:          {train_args.encounters or 'all'}", flush=True)
    if train_args.limit is not None:
        print(f"limit:               {train_args.limit}", flush=True)

    checkpoint = train_value(train_args)
    checkpoint_path = train_args.output
    checkpoint_json = checkpoint_path.with_suffix(".json")

    export_cfg = config.get("export", {})
    exported = None
    if bool(export_cfg.get("native_weights", True)):
        exported = out / str(export_cfg.get("path", "value_weights.bin"))
        print("\nExport", flush=True)
        print("------", flush=True)
        print(f"native weights: {exported}", flush=True)
        export_weights(checkpoint_path, exported)

    write_summary(out, checkpoint_path, checkpoint_json, train_args.data, exported)
    print("\nOutputs", flush=True)
    print("-------", flush=True)
    print(f"checkpoint:      {checkpoint_path}", flush=True)
    print(f"checkpoint json: {checkpoint_json}", flush=True)
    if exported:
        print(f"native weights:  {exported}", flush=True)
    print(f"summary:         {out / 'summary.json'}", flush=True)
    print("\nDone", flush=True)
    print(f"validation_mse: {checkpoint['metrics'].get('validation_mse')}", flush=True)
    return 0


if __name__ == "__main__":
    main(train, inputs)
