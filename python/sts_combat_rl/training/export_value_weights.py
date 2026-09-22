"""Write a value checkpoint's weights in the flat binary format read by models/value_net.cpp.

Layout (little endian): b"STSVNET1", u32 config length, config JSON (architecture,
encoding_version), u32 tensor count, then per tensor: u32 name length, name, u32 ndim,
u32 dims[ndim], float32 data in row-major order.

    python -m sts_combat_rl.training.export_value_weights runs/x/value_checkpoint.pt runs/x/value_weights.bin
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch


def export(checkpoint_path: Path, output: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("encoding_version") != 3:
        raise ValueError("checkpoint is not encoding v3")
    config = json.dumps(
        {"architecture": checkpoint["architecture"], "encoding_version": 3}, sort_keys=True
    ).encode()
    state = checkpoint["model_state"]
    with output.open("wb") as f:
        f.write(b"STSVNET1")
        f.write(struct.pack("<I", len(config)))
        f.write(config)
        f.write(struct.pack("<I", len(state)))
        for name, tensor in state.items():
            data = tensor.detach().to(torch.float32).contiguous().numpy().astype("<f4")
            encoded = name.encode()
            f.write(struct.pack("<I", len(encoded)))
            f.write(encoded)
            f.write(struct.pack(f"<I{data.ndim}I", data.ndim, *data.shape))
            f.write(data.tobytes())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
