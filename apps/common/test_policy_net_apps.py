"""App plumbing for leaf policy_net and skip_diverged (apps/common/app.py, apps/common/replay.py)."""
import unittest

from apps.common.app import teacher_settings
from apps.common.replay import diverged

BASE = {"leaf": "policy_net", "simulations": 100, "oracle": False, "random_move": False, "particles": 8,
        "c_puct": 1.0, "fpu_reduction": 0.05, "prior_floor": 0.03}


class PolicyNetAppTests(unittest.TestCase):
    def test_policy_settings_required_with_policy_net_only(self):
        self.assertEqual(teacher_settings(dict(BASE))["c_puct"], 1.0)
        with self.assertRaises(SystemExit):
            teacher_settings({k: v for k, v in BASE.items() if k != "prior_floor"})
        with self.assertRaises(SystemExit):
            teacher_settings({**BASE, "leaf": "value_net"})

    def test_diverged_only_matches_replay_divergence(self):
        self.assertTrue(diverged(RuntimeError("value_play_worker exited 1 on request {...}: stored actions don't fit "
                                              "the replayed fight")))
        self.assertTrue(diverged(RuntimeError("episode 5: replayed start differs from the stored fight on ['cards']")))
        self.assertFalse(diverged(RuntimeError("worker exited 1: non-finite leaf value")))
        self.assertFalse(diverged(ValueError("stored actions don't fit the replayed fight")))


if __name__ == "__main__":
    unittest.main()
