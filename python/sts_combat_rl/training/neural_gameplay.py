"""Persistent framed MessagePack controller for neural MCTS gameplay."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path
from typing import BinaryIO

import msgpack
import torch

from sts_combat_rl.models.deep_sets import DeepSetsValue

from .data import collate_states


def exact_read(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise RuntimeError("gameplay process closed protocol stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def load_model(path: Path) -> DeepSetsValue:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("encoding_version") != 3
        or checkpoint.get("target_name") != "mcts_value"
    ):
        raise ValueError("checkpoint is not schema-v3 mcts_value")
    model = DeepSetsValue(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def play(binary: Path, model: DeepSetsValue, seed: int, simulations: int) -> dict:
    process = subprocess.Popen(
        [str(binary), str(seed), str(simulations)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    try:
        while True:
            size = struct.unpack("!I", exact_read(process.stdout, 4))[0]
            message = msgpack.unpackb(exact_read(process.stdout, size), raw=False)
            if message["type"] == "result":
                if process.wait() != 0:
                    raise RuntimeError("gameplay process failed")
                return message
            if message["type"] != "leaf":
                raise ValueError("unknown gameplay protocol message")
            with torch.no_grad():
                batch = collate_states([message["state"]])
                batch.pop("target")
                value = float(model(**batch).item())
            if not torch.isfinite(torch.tensor(value)) or not -1 <= value <= 1:
                raise ValueError(f"invalid prediction {value}")
            process.stdin.write(struct.pack("!f", value))
            process.stdin.flush()
    except Exception:
        process.kill()
        process.wait()
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("seed", type=int)
    parser.add_argument("--simulations", type=int, default=20)
    parser.add_argument(
        "--binary", type=Path, default=Path("build/main/play_slime_mcts_neural")
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(
        json.dumps(
            play(args.binary, load_model(args.checkpoint), args.seed, args.simulations)
        )
    )


if __name__ == "__main__":
    main()
