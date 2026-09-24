"""Orchestration checks for apps/dagger/generate.py (no fights played)."""
import unittest

from apps.dagger import generate
from test_value_play import fake_runs


class SourceRunTests(unittest.TestCase):
    def test_bootstrap_inputs_of_dagger_value_run(self):
        runs = {"value_net_v1/d/gen1": ["value_net_v1/d/gen0", "combat_v3/d/boot", "combat_v3/d/dagger"],
                "value_net_v1/d/gen0": ["combat_v3/d/boot"], "combat_v3/d/boot": [], "combat_v3/d/dagger": ["value_net_v1/d/gen0"]}
        with fake_runs(runs):
            self.assertEqual(generate.inputs({"run": {"input": "value_net_v1/d/gen1"}}),
                             ["value_net_v1/d/gen1", "combat_v3/d/boot"])


if __name__ == "__main__":
    unittest.main()
