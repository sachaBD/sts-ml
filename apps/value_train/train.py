#!/usr/bin/env python3
"""Config-driven value-network training app.

A thin TOML wrapper around sts_combat_rl.training.train_value for the run launcher (apps/common/launch.sh).
Every setting is explicit (train_value.TrainConfig): [data] selects the combat_v3 rows (sts_combat_rl.query),
[model] is the architecture (sts_combat_rl.topology.build: kind and every argument), [train] the rest. Writes
value_checkpoint.pt (+ .json), the native C++ weights value_weights.bin, and summary.json for run.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from apps.common.app import check_keys, main, required, value_run, write_json
from sts_combat_rl import query
from sts_combat_rl.training.export_value_weights import export as export_weights
from sts_combat_rl.training.train_value import TrainConfig, run as train_value

TABLES = {"run", "data", "model", "train"}
DATA_REQUIRED, DATA_OPTIONAL = {"query", "oracle"}, {"corrections"}
TRAIN_REQUIRED = {"epochs", "batch_size", "lr", "weight_decay", "seed", "label", "lr_schedule", "keep", "threads",
                  "device", "aux_won_weight", "aux_hp_weight"}
# Required exactly when they apply (TrainConfig checks which): absent means off / not applicable.
TRAIN_OPTIONAL = {"blend", "initial_checkpoint", "split", "validation_fraction", "correction_weight", "row_weighting",
                  "aux_keep_weight", "policy_weight"}
SUMMARY_KEYS = ("initial_checkpoint", "initial_checkpoint_sha256", "initialization", "split",
                "training_kind", "correction_weight", "correction_target", "correction_rows", "correction_query", "correction_source")


def inputs(config: dict[str, Any]) -> list[str]:
    """Every run this trains from: the initial checkpoint's run, the data runs, the correction runs."""
    data, initial = config["data"], config["train"].get("initial_checkpoint")
    return [*([initial] if initial else []), *query.run_ids(data["query"]),
            *(query.run_ids(data["corrections"]) if "corrections" in data else [])]


def initial_checkpoint(value: str | None) -> Path | None:
    """[train].initial_checkpoint: a value_net_v1 run_id (-> its finished checkpoint) or a checkpoint path."""
    if value is None:
        return None
    return Path(value) if Path(value).is_file() else value_run(value).checkpoint


def make_train_config(config: dict[str, Any], out: Path) -> TrainConfig:
    check_keys(config, TABLES, "top level")
    data, model, train = (required(config, name, "top level") for name in ("data", "model", "train"))
    check_keys(data, DATA_REQUIRED | DATA_OPTIONAL, "data")
    check_keys(train, TRAIN_REQUIRED | TRAIN_OPTIONAL, "train")
    get = lambda table, name, key: required(table, key, name)
    return TrainConfig(
        data=get(data, "data", "query"),
        oracle=get(data, "data", "oracle"),
        corrections=data.get("corrections"),
        model=dict(model),
        output=out / "value_checkpoint.pt",
        initial_checkpoint=initial_checkpoint(train.get("initial_checkpoint")),
        **{key: get(train, "train", key) for key in TRAIN_REQUIRED},
        **{key: train.get(key) for key in TRAIN_OPTIONAL - {"initial_checkpoint"}},
    )


def write_summary(out: Path, checkpoint_path: Path, checkpoint_json: Path, exported: Path) -> None:
    metadata = json.loads(checkpoint_json.read_text())
    summary = {
        "checkpoint": checkpoint_path.name,
        "checkpoint_json": checkpoint_json.name,
        "weights": exported.name,
        "data_query": metadata["data_query"],
        "source_runs": metadata["source_runs"],
        "data_schema": metadata["data_schema"],
        "row_counts": metadata["row_counts"],
        "target_name": metadata["target_name"],
        "target_blend": metadata["target_blend"],
        "architecture": metadata["architecture"],
        "epoch": metadata["epoch"],
        "metrics": metadata["metrics"],
        "split_mode": metadata["split_mode"],
        "train_episodes": len(metadata["train_episode_ids"]),
        "validation_episodes": len(metadata["validation_episode_ids"]),
        "train_runs": len(metadata["train_run_seeds"]),
        "validation_runs": len(metadata["validation_run_seeds"]),
        **{key: metadata[key] for key in SUMMARY_KEYS if key in metadata},
    }
    if "correction_episode_ids" in metadata:
        summary["correction_episodes"] = len(metadata["correction_episode_ids"])
    write_json(out / "summary.json", summary)


def train(config: dict[str, Any], config_path: Path, out: Path) -> int:
    args = make_train_config(config, out)
    for line in ("", "Value training", "==============", f"config:     {config_path}", f"output:     {out}",
                 f"checkpoint: {args.output}", "", "Inputs", "------", f"data:        {args.data}",
                 *([f"corrections: {args.corrections}"] if args.corrections else []),
                 *([f"initial:     {args.initial_checkpoint}"] if args.initial_checkpoint else []),
                 *(["oracle rows allowed"] if args.oracle else []),
                 "", "Model", "-----", *(f"{k + ':':<21}{v}" for k, v in args.model.items()),
                 "", "Config", "------",
                 *(f"{key + ':':<21}{value}" for key, value in vars(args).items()
                   if key not in ("data", "corrections", "initial_checkpoint", "oracle", "model", "output"))):
        print(line, flush=True)

    checkpoint = train_value(args)
    checkpoint_json = args.output.with_suffix(".json")
    exported = out / "value_weights.bin"
    print(f"\nExport\n------\nnative weights: {exported}", flush=True)
    export_weights(args.output, exported)

    write_summary(out, args.output, checkpoint_json, exported)
    for line in ("", "Outputs", "-------", f"checkpoint:      {args.output}", f"checkpoint json: {checkpoint_json}",
                 f"native weights:  {exported}", f"summary:         {out / 'summary.json'}",
                 "", "Done", f"validation_mse: {checkpoint['metrics']['validation_mse']}"):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    main(train, inputs)
