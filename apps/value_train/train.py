#!/usr/bin/env python3
"""Config-driven value-network training app.

A thin TOML wrapper around sts_combat_rl.training.train_value for the run launcher (apps/common/launch.sh):
[data] query selects the combat_v3 rows (sts_combat_rl.query); writes a checkpoint into --out, optionally
exports native C++ weights, and emits summary.json for run.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from apps.common.app import check_keys, flag, main, value_run, write_json
from sts_combat_rl import query
from sts_combat_rl.training.export_value_weights import export as export_weights
from sts_combat_rl.training.train_value import run as train_value

DATA_KEYS = {"query", "corrections", "oracle"}
TRAIN_KEYS = {"initial_checkpoint", "split", "correction_weight", "checkpoint", "epochs", "batch_size", "lr", "weight_decay",
              "width", "seed", "validation_fraction", "label", "blend"}
SUMMARY_KEYS = ("initial_checkpoint", "initial_checkpoint_sha256", "initialization", "split_mode", "split",
                "training_kind", "correction_weight", "correction_target", "correction_rows", "correction_query", "correction_source")


def inputs(config: dict[str, Any]) -> list[str]:
    """Every run this trains from: the initial checkpoint's run, the data runs, the correction runs."""
    data, initial = config["data"], config.get("train", {}).get("initial_checkpoint")
    return [*([initial] if initial else []), *query.run_ids(data["query"]),
            *(query.run_ids(data["corrections"]) if "corrections" in data else [])]


def initial_checkpoint(value: str | None) -> Path | None:
    """[train].initial_checkpoint: a value_net_v1 run_id (-> its finished checkpoint) or a checkpoint path."""
    if value is None:
        return None
    return Path(value) if Path(value).is_file() else value_run(value).checkpoint


def make_train_args(config: dict[str, Any], out: Path) -> SimpleNamespace:
    data, train = config["data"], config.get("train", {})
    check_keys(data, DATA_KEYS, "data")
    check_keys(train, TRAIN_KEYS, "train")
    return SimpleNamespace(
        data=data["query"],
        corrections=data.get("corrections"),
        oracle=flag(data, "oracle"),
        initial_checkpoint=initial_checkpoint(train.get("initial_checkpoint")),
        # with initial_checkpoint: "pinned" to its train/validation episodes, or "fresh" episode_split of the data
        split=str(train.get("split", "pinned")),
        correction_weight=float(train.get("correction_weight", 0.5)),
        output=out / str(train.get("checkpoint", "value_checkpoint.pt")),
        epochs=int(train.get("epochs", 20)),
        batch_size=int(train.get("batch_size", 128)),
        lr=float(train.get("lr", 0.001)),
        weight_decay=float(train.get("weight_decay", 0.0001)),
        width=int(train.get("width", 64)),
        seed=int(train.get("seed", 0)),
        validation_fraction=float(train.get("validation_fraction", 0.2)),
        label=str(train.get("label", "blend")),  # blend, root or terminal
        blend=float(train.get("blend", 0.5)),
    )


def write_summary(out: Path, checkpoint_path: Path, checkpoint_json: Path, exported: Path | None) -> None:
    metadata = json.loads(checkpoint_json.read_text())
    summary = {
        "checkpoint": checkpoint_path.name,
        "checkpoint_json": checkpoint_json.name,
        "weights": exported.name if exported else None,
        "data_query": metadata["data_query"],
        "source_runs": metadata["source_runs"],
        "data_schema": metadata["data_schema"],
        "row_counts": metadata["row_counts"],
        "target_name": metadata.get("target_name"),
        "target_blend": metadata.get("target_blend"),
        "epoch": metadata.get("epoch"),
        "metrics": metadata.get("metrics"),
        "train_episodes": len(metadata.get("train_episode_ids", [])),
        "validation_episodes": len(metadata.get("validation_episode_ids", [])),
        "train_runs": len(metadata["train_run_seeds"]),
        "validation_runs": len(metadata["validation_run_seeds"]),
        **{key: metadata[key] for key in SUMMARY_KEYS if key in metadata},
    }
    if "correction_episode_ids" in metadata:
        summary["correction_episodes"] = len(metadata["correction_episode_ids"])
    write_json(out / "summary.json", summary)


def train(config: dict[str, Any], config_path: Path, out: Path) -> int:
    args = make_train_args(config, out)
    for line in ("", "Value training", "==============", f"config:     {config_path}", f"output:     {out}",
                 f"checkpoint: {args.output}", "", "Inputs", "------", f"data:        {args.data}",
                 *([f"corrections: {args.corrections}"] if args.corrections else []),
                 *([f"initial:     {args.initial_checkpoint}"] if args.initial_checkpoint else []),
                 *(["oracle rows allowed"] if args.oracle else []), "", "Config", "------",
                 *(f"{key + ':':<21}{getattr(args, key)}" for key in ("epochs", "batch_size", "lr", "weight_decay", "width",
                                                                     "seed", "validation_fraction", "label", "blend", "split"))):
        print(line, flush=True)

    checkpoint = train_value(args)
    checkpoint_json = args.output.with_suffix(".json")
    export = config.get("export", {})
    exported = None
    if bool(export.get("native_weights", True)):
        exported = out / str(export.get("path", "value_weights.bin"))
        print(f"\nExport\n------\nnative weights: {exported}", flush=True)
        export_weights(args.output, exported)

    write_summary(out, args.output, checkpoint_json, exported)
    for line in ("", "Outputs", "-------", f"checkpoint:      {args.output}", f"checkpoint json: {checkpoint_json}",
                 *([f"native weights:  {exported}"] if exported else []), f"summary:         {out / 'summary.json'}",
                 "", "Done", f"validation_mse: {checkpoint['metrics'].get('validation_mse')}"):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    main(train, inputs)
