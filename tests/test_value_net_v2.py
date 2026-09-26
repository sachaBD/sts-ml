"""DeepSetsValueV2: build_model / v1 compatibility, and C++ parity (value_net_v2_test binary, if built).

The binary is $VALUE_NET_V2_TEST (set by ctest), else build/main/value_net_v2_test.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from sts_combat_rl.models import DeepSetsValue, DeepSetsValueV2, build_model  # noqa: E402
from sts_combat_rl.training.data import collate_states  # noqa: E402
from test_value_training import row  # noqa: E402

import value_net_v2_golden  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("VALUE_NET_V2_TEST", ROOT / "build/main/value_net_v2_test"))


def batch(rows):
    b = collate_states(rows)
    b.pop("target")
    return b


class BuildModelTests(unittest.TestCase):
    def test_v1_unchanged(self):
        torch.manual_seed(0)
        a = DeepSetsValue(width=64)
        torch.manual_seed(0)
        b = build_model({"width": 64})
        self.assertIsInstance(b, DeepSetsValue)
        self.assertEqual(b.config, a.config)
        self.assertNotIn("kind", b.config)
        for (na, pa), (nb, pb) in zip(a.state_dict().items(), b.state_dict().items()):
            self.assertEqual(na, nb)
            self.assertTrue(torch.equal(pa, pb))
        x = batch([row(0, 2), row(1, 3)])
        self.assertTrue(torch.equal(a(**x), b.forward_all(**x)["value"]))

    def test_v2_config_roundtrip_and_errors(self):
        model = build_model({"kind": "deep_sets_v2", "head_width": 64})
        self.assertIsInstance(model, DeepSetsValueV2)
        self.assertEqual(model.config["kind"], "deep_sets_v2")
        again = build_model(model.config)
        again.load_state_dict(model.state_dict())
        self.assertEqual(again.config, model.config)
        with self.assertRaises(ValueError):
            build_model({"kind": "deep_sets_v2", "head_widht": 64})
        with self.assertRaises(ValueError):
            build_model({"kind": "transformer"})

    def test_v2_outputs_and_batching(self):
        torch.manual_seed(1)
        model = build_model({"kind": "deep_sets_v2"})
        rows = [row(0, 2), row(1, 3)]
        rows[1]["cards"].append(dict(rows[1]["cards"][0], zone=4))  # offered: not pooled
        out = model.forward_all(**batch(rows))
        self.assertEqual(set(out), {"value", "won_logit", "hp"})
        self.assertTrue(((out["value"] > 0) & (out["value"] < 1)).all())
        singles = torch.cat([model(**batch([r])) for r in rows])
        self.assertTrue(torch.allclose(out["value"], singles, atol=1e-6))
        plain = build_model({"kind": "deep_sets_v2", "aux_heads": False, "output": "tanh", "head_blocks": 0})
        self.assertEqual(set(plain.forward_all(**batch(rows))), {"value"})
        self.assertFalse(any(n.startswith(("won_out", "hp_out", "head_out_norm")) for n in plain.state_dict()))


@unittest.skipUnless(BINARY.exists(), f"{BINARY} not built")
class CppParityTests(unittest.TestCase):
    def test_parity(self):
        with tempfile.TemporaryDirectory() as out:
            value_net_v2_golden.golden(Path(out))
            result = subprocess.run([str(BINARY), out, *value_net_v2_golden.CONFIGS], capture_output=True, text=True)
            print(result.stdout, end="")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
