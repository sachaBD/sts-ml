"""Paired neural-versus-rollout evaluation on held-out deck+HP roots."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import time
from pathlib import Path

import msgpack
import torch

from .data import collate_states
from .neural_gameplay import exact_read, load_model

EVAL_SEED_XOR = 0xA0761D6478BD642F  # disjoint from entry training replicate seeds


def neural(binary, source, selector, seed, simulations, model):
    process = subprocess.Popen(
        [str(binary), str(source), selector, str(seed), str(simulations)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin and process.stdout
    try:
        while True:
            size = struct.unpack("!I", exact_read(process.stdout, 4))[0]
            message = msgpack.unpackb(exact_read(process.stdout, size), raw=False)
            if message["type"] == "result":
                if process.wait() != 0:
                    raise RuntimeError("neural gameplay process failed")
                return message
            with torch.no_grad():
                batch = collate_states([message["state"]])
                batch.pop("target")
                value = float(model(**batch).item())
            process.stdin.write(struct.pack("!f", value))
            process.stdin.flush()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def rollout(binary, source, selector, seed, simulations):
    return json.loads(
        subprocess.check_output(
            [str(binary), str(source), selector, str(seed), str(simulations)], text=True
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--neural", type=Path, default=Path("build/play_entry_mcts_neural"))
    p.add_argument(
        "--rollout", type=Path, default=Path("build/play_entry_mcts_rollout")
    )
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = load_model(args.checkpoint)
    signatures = set(checkpoint["validation_deck_signatures"])
    roots = []
    for line in args.source.read_text().splitlines():
        row = json.loads(line)
        if row.get("status") == "accepted" and row["deck_signature"] in signatures:
            roots.append(row)
    if len(roots) != len(signatures):
        raise ValueError("held-out signature/root mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = args.output / "episodes.jsonl"
    if episodes.exists():
        raise FileExistsError(episodes)
    start = time.monotonic()
    pairs = []
    with episodes.open("x") as out:
        for i, row in enumerate(roots, 1):
            seed = int(row["seed"]) ^ EVAL_SEED_XOR
            for mode in ("rollout", "neural"):
                result = (
                    rollout(args.rollout, args.source, row["deck_signature"], seed, 100)
                    if mode == "rollout"
                    else neural(
                        args.neural,
                        args.source,
                        row["deck_signature"],
                        seed,
                        100,
                        model,
                    )
                )
                result.update(
                    {
                        "mode": mode,
                        "evaluation_seed_contract": "source_seed xor 0xa0761d6478bd642f",
                        "projection": "deck_hp_only",
                    }
                )
                out.write(json.dumps(result, sort_keys=True) + "\n")
                out.flush()
                if mode == "rollout":
                    base = result
                else:
                    pairs.append((base, result))
            print(f"[{i}/{len(roots)}] {row['deck_signature']}", flush=True)

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "roots": len(roots),
        "pairs": len(pairs),
        "elapsed_seconds": time.monotonic() - start,
        "rollout_wins": sum(x[0]["won"] for x in pairs),
        "neural_wins": sum(x[1]["won"] for x in pairs),
        "rollout_mean_final_hp": mean([x[0]["final_hp"] for x in pairs]),
        "neural_mean_final_hp": mean([x[1]["final_hp"] for x in pairs]),
        "neural_minus_rollout_final_hp": mean(
            [x[1]["final_hp"] - x[0]["final_hp"] for x in pairs]
        ),
        "rescued_wins": sum(not x[0]["won"] and x[1]["won"] for x in pairs),
        "lost_wins": sum(x[0]["won"] and not x[1]["won"] for x in pairs),
        "rollout_decisions": sum(x[0]["decisions"] for x in pairs),
        "neural_decisions": sum(x[1]["decisions"] for x in pairs),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
