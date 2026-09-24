"""Orchestration checks for apps/dagger/generate.py (no fights played)."""
import importlib.util
import unittest
from pathlib import Path

from test_value_play import fake_runs

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dagger_generate", ROOT / "apps/dagger/generate.py")
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)


class SourceRunTests(unittest.TestCase):
    def test_bootstrap_inputs_of_dagger_value_run(self):
        runs = {"value_net_v1/d/gen1": ["value_net_v1/d/gen0", "combat_v3/d/boot", "combat_v3/d/dagger"],
                "value_net_v1/d/gen0": ["combat_v3/d/boot"], "combat_v3/d/boot": [], "combat_v3/d/dagger": ["value_net_v1/d/gen0"]}
        with fake_runs(runs):
            self.assertEqual(generate.source_runs("value_net_v1/d/gen1"), ["combat_v3/d/boot"])


if __name__ == "__main__":
    unittest.main()
