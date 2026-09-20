#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sts_combat_rl import DeepSetsValue, value_tensors

root = Path(__file__).resolve().parents[1]
dump = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "build" / "dump_slime_boss_encoding"
raw = subprocess.check_output([str(dump), "1"], text=True)
state = json.loads(raw)
print(json.dumps(state, indent=2))
tensors = value_tensors(state)
torch.manual_seed(0)
model = DeepSetsValue()
print(model)
print({name: tuple(value.shape) for name, value in tensors.items()})
print("value:", model(**tensors).item())
