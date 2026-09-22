"""Native C++ value net (build/value_net_eval) matches PyTorch on real encoded states.

Needs build/value_net_eval, runs/gen0b-value-v1/value_checkpoint.pt and the gen0b data run;
skipped when any is missing. Run from the repo root:

    PYTHONPATH=python .venv/bin/python3 -m unittest tests.test_value_net_parity
"""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from sts_combat_rl.models.deep_sets import DeepSetsValue
from sts_combat_rl.training.data import collate_states, load_rows
from sts_combat_rl.training.export_value_weights import export

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/value_net_eval"
CHECKPOINT = ROOT / "runs/gen0b-value-v1/value_checkpoint.pt"
DATA = ROOT / "data/combat/run_id=2026-09-22_slime_pbcs15k_gen0b"
STATE_KEYS = (
    "global_numeric",
    "input_state",
    "card_selection_task",
    "cards",
    "monsters",
    "card_monster_interactions",
)
STATES = 2000
TOLERANCE = 1e-5


@unittest.skipUnless(
    BINARY.exists() and CHECKPOINT.exists() and DATA.exists(), "needs build, checkpoint and data"
)
class ValueNetParityTest(unittest.TestCase):
    def test_native_matches_pytorch(self):
        rows = load_rows(DATA)
        rows = rows[:: max(1, len(rows) // STATES)][:STATES]  # spread over the whole run
        states = [{k: r[k] for k in STATE_KEYS} for r in rows]
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model = DeepSetsValue(**checkpoint["architecture"])
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        with torch.no_grad():
            batch = collate_states(states)
            batch.pop("target")
            expected = model(**batch).reshape(-1)
        with tempfile.TemporaryDirectory() as tmp:
            weights, inputs = Path(tmp) / "value_weights.bin", Path(tmp) / "states.jsonl"
            export(CHECKPOINT, weights)
            inputs.write_text("".join(json.dumps(s) + "\n" for s in states))
            output = subprocess.run(
                [BINARY, weights, inputs], check=True, capture_output=True, text=True
            ).stdout
        native = torch.tensor([float(x) for x in output.split()])
        self.assertEqual(len(native), len(states))
        difference = (native - expected).abs().max().item()
        print(f"parity: {len(states)} states, max |native - torch| = {difference:.3g}")
        self.assertLess(difference, TOLERANCE)


if __name__ == "__main__":
    unittest.main()
