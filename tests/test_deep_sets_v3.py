"""deep_sets_v3: the score-formula output, and Python / C++ (topology/value_net.cpp) parity on v4 encodings.

The parity test runs build/<STSRL_BUILD_DIR, default main>/encoding_v4_test; it is skipped if that is not built.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from sts_combat_rl.topology import ENCODING_VERSIONS, build
from sts_combat_rl.training.encoding_v4 import collate_v4
from sts_combat_rl.training.export_value_weights import export

ROOT = Path(__file__).resolve().parents[1]
V3 = {"kind": "deep_sets_v3", "card_vocab": 512, "monster_vocab": 128, "move_vocab": 512, "potion_vocab": 64,
      "relic_vocab": 192, "width": 64, "card_id_dim": 16, "monster_id_dim": 8, "move_dim": 8, "potion_id_dim": 8,
      "relic_id_dim": 8, "id_dropout": 0.15, "pool_count_features": True, "head_input_norm": True,
      "head_width": 64, "head_blocks": 2, "zero_init_blocks": False, "score_hp_offset": 35.0,
      "score_potion_hp": 4.0, "score_max_hp_offset": 55.0}


def model(seed=0):
    torch.manual_seed(seed)
    net = build(V3)
    with torch.no_grad():  # larger random output heads so the score terms all matter
        for head in (net.won_out, net.hp_out, net.keep_out):
            head.weight.normal_(0, 0.5)
            head.bias.normal_(0, 0.5)
    return net.eval()


def c_values(net):
    binary = ROOT / "build" / os.environ.get("STSRL_BUILD_DIR", "main") / "encoding_v4_test"
    if not binary.exists():
        raise unittest.SkipTest(f"{binary} is not built")
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint, weights, out = Path(tmp, "c.pt"), Path(tmp, "w.bin"), Path(tmp, "out.json")
        torch.save({"architecture": net.config, "encoding_version": ENCODING_VERSIONS["deep_sets_v3"],
                    "model_state": net.state_dict()}, checkpoint)
        export(checkpoint, weights)
        subprocess.run([str(binary), str(weights), str(out)], check=True)
        return json.loads(out.read_text())


class DeepSetsV3Tests(unittest.TestCase):
    def test_parity_with_cpp(self):
        net = model()
        rows = c_values(net)
        states = [r["state"] for r in rows]
        self.assertTrue(any(s["potions"] for s in states) and any(s["relics"] for s in states))
        with torch.no_grad():
            values = net(**collate_v4(states))
        # one batch and one state at a time (pooling must not mix states)
        for i, r in enumerate(rows):
            self.assertAlmostEqual(values[i].item(), r["value"], places=5)
            with torch.no_grad():
                self.assertAlmostEqual(net(**collate_v4([states[i]])).item(), r["value"], places=5)

    def test_score_formula(self):
        net = model(1)
        rows = c_values(net)
        batch = collate_v4([r["state"] for r in rows])
        with torch.no_grad():
            out = net.forward_all(**batch)
        potions = torch.bincount(batch["potion_state_indices"], minlength=len(rows)).float()
        max_hp = batch["max_hp"]
        expected = torch.sigmoid(out["won_logit"]) * (35 + out["hp_fraction"] * max_hp
                                                      + 4 * out["keep_fraction"] * potions) / (55 + max_hp)
        torch.testing.assert_close(out["value"], expected)
        self.assertTrue(((out["value"] >= 0) & (out["value"] <= 1)).all())

    def test_id_dropout_only_in_training(self):
        net = model(2)
        batch = collate_v4([r["state"] for r in c_values(net)])
        with torch.no_grad():
            a, b = net(**batch), net(**batch)
        torch.testing.assert_close(a, b)
        net.train()
        net.id_dropout = 1.0
        ids = batch["potion_ids"]
        self.assertTrue((net._drop_ids(ids) == 0).all())

    def test_architecture_is_explicit(self):
        with self.assertRaises(ValueError):
            build({k: v for k, v in V3.items() if k != "id_dropout"})


if __name__ == "__main__":
    unittest.main()
