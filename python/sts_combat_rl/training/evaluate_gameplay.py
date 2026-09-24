"""Run rollout and neural Slime Boss gameplay on identical seeds."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch

from .neural_gameplay import load_model, play


def aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    observed_splits = [row["split_hp"] for row in rows if row["split_hp"] >= 0]
    return {
        "episodes": count,
        "win_rate": sum(row["won"] for row in rows) / count,
        "final_hp": sum(row["final_hp"] for row in rows) / count,
        "split_hp": sum(observed_splits) / len(observed_splits)
        if observed_splits
        else None,
        "split_count": len(observed_splits),
        "decisions": sum(row["decisions"] for row in rows) / count,
        "elapsed_seconds": sum(row["elapsed_seconds"] for row in rows) / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--first-seed", type=int, default=1001)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=10)
    parser.add_argument(
        "--neural-binary", type=Path, default=Path("build/main/play_slime_mcts_neural")
    )
    parser.add_argument(
        "--rollout-binary", type=Path, default=Path("build/main/play_slime_mcts_rollout")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_model(args.checkpoint)
    results: dict[str, list[dict]] = {"neural": [], "rollout": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for index, seed in enumerate(
            range(args.first_seed, args.first_seed + args.count), 1
        ):
            neural = play(args.neural_binary, model, seed, args.simulations)
            neural["mode"] = "neural"
            results["neural"].append(neural)
            output.write(json.dumps(neural) + "\n")
            output.flush()
            print(
                f"progress mode=neural episode={index}/{args.count} seed={seed}",
                flush=True,
            )

            rollout = json.loads(
                subprocess.check_output(
                    [str(args.rollout_binary), str(seed), str(args.simulations)],
                    text=True,
                )
            )
            rollout["mode"] = "rollout"
            results["rollout"].append(rollout)
            output.write(json.dumps(rollout) + "\n")
            output.flush()
            print(
                f"progress mode=rollout episode={index}/{args.count} seed={seed}",
                flush=True,
            )
    print(json.dumps({mode: aggregate(rows) for mode, rows in results.items()}))


if __name__ == "__main__":
    main()
