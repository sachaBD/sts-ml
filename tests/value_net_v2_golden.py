#!/usr/bin/env python3
"""Golden outputs of DeepSetsValueV2 for the C++ parity test (tests/value_net_v2_test.cpp).

    value_net_v2_golden.py OUT_DIR              per config: OUT_DIR/<name>.bin (export_value_weights format) and
                                                OUT_DIR/<name>.json {"values": [...]} over tests/data/value_net_v2_states.json
    value_net_v2_golden.py --refresh-states     re-sample those states from the combat_v3 dataset

Weights are random (seeded) with every parameter perturbed, so LayerNorm affine and residual paths are exercised.
tests/test_value_net_v2.py runs this and the C++ test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sts_combat_rl.models import build_model
from sts_combat_rl.training.data import collate_states
from sts_combat_rl.training.export_value_weights import export

STATES = Path(__file__).parent / "data" / "value_net_v2_states.json"
DATASET = "select * from combat_v3 where id = 'act1-all-bosses-a20-scaled-search' and run_seed % 997 = 0 limit 20000"
COLUMNS = ["global_numeric", "input_state", "card_selection_task", "cards", "monsters", "card_monster_interactions"]
CONFIGS = {
    "default": {"kind": "deep_sets_v2"},
    "small_head": {"kind": "deep_sets_v2", "head_width": 128, "head_blocks": 1},
    "width64": {"kind": "deep_sets_v2", "width": 64},
    "plain_tanh": {"kind": "deep_sets_v2", "aux_heads": False, "head_blocks": 0, "output": "tanh",
                   "head_input_norm": False, "pool_count_features": False, "head_width": 96},
    "blocks0_tanh": {"kind": "deep_sets_v2", "aux_heads": False, "head_blocks": 0, "output": "tanh"},
}


def refresh_states(count: int = 48) -> None:
    from sts_combat_rl import query
    rows = query.rows(DATASET, columns=COLUMNS)
    # spread over the sample, plus the rows with the most cards / interactions
    picked = {round(i * (len(rows) - 1) / (count - 5)) for i in range(count - 4)}
    picked |= set(sorted(range(len(rows)), key=lambda i: -len(rows[i]["cards"]))[:2])
    picked |= set(sorted(range(len(rows)), key=lambda i: -len(rows[i]["card_monster_interactions"]))[:2])
    states = [{key: rows[i][key] for key in COLUMNS} for i in sorted(picked)]
    # synthetic edge cases: an offered card (zone 4, not pooled / counted), another input state; no interactions
    offered = json.loads(json.dumps(states[0]))
    offered["cards"][-1]["zone"] = 4
    offered["input_state"], offered["card_selection_task"] = 2, 1
    bare = json.loads(json.dumps(states[1]))
    bare["card_monster_interactions"] = []
    states += [offered, bare]
    STATES.parent.mkdir(parents=True, exist_ok=True)
    STATES.write_text(json.dumps(states) + "\n")
    print(f"{len(states)} states -> {STATES}")


def golden(out: Path) -> None:
    states = json.loads(STATES.read_text())
    batch = collate_states(states)
    batch.pop("target")
    out.mkdir(parents=True, exist_ok=True)
    for seed, (name, architecture) in enumerate(CONFIGS.items()):
        torch.manual_seed(seed)
        model = build_model(architecture)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
        model.eval()
        with torch.no_grad():
            values = model(**batch)
        checkpoint = out / f"{name}.pt"
        torch.save({"model_state": model.state_dict(), "architecture": model.config, "encoding_version": 3}, checkpoint)
        export(checkpoint, out / f"{name}.bin")
        checkpoint.unlink()
        (out / f"{name}.json").write_text(json.dumps({"architecture": model.config, "values": values.tolist()}))
        print(f"{name}: {sum(p.numel() for p in model.parameters()):,} params, values {values.min():.4f}..{values.max():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, nargs="?")
    parser.add_argument("--refresh-states", action="store_true")
    args = parser.parse_args()
    if args.refresh_states:
        refresh_states()
    if args.out:
        golden(args.out)


if __name__ == "__main__":
    main()
