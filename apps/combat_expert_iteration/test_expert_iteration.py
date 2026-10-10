import unittest

from apps.combat_expert_iteration.expert_iteration import make_starts, score


class ExpertIterationTest(unittest.TestCase):
    def test_starts_are_deterministic_and_never_reuse_seeds(self):
        base = {"fight_id": "x", "seed_kind": "human", "start": {"seed": 1, "misc_rng": None, "potion_rng": None}}
        used = {1}
        a = make_starts(base, "ns", "train", 5, used)
        self.assertEqual([r["start"]["seed"] for r in a], [r["start"]["seed"] for r in make_starts(base, "ns", "train", 5, set())])
        b = make_starts(base, "ns", "monitor", 5, used)
        seeds = [r["start"]["seed"] for r in a + b]
        self.assertEqual(len(set(seeds)), 10)
        self.assertNotIn(1, seeds)
        self.assertEqual(base["start"]["seed"], 1)  # base row untouched

    def test_score_pairs_only_fights_both_completed(self):
        won = lambda w: {"status": "completed", "fight": {"won": w}}
        reference = {"r:x:m:0": won(False), "r:x:m:1": won(True), "r:x:m:2": won(True), "r:y:m:0": won(True)}
        candidate = {"r:x:m:0": won(True), "r:x:m:1": won(True), "r:x:m:2": {"status": "error"}, "r:y:m:0": won(False)}
        s = score(reference, candidate)
        self.assertEqual((s["n"], s["wins"], s["teacher_wins"]), (3, 2, 2))
        self.assertAlmostEqual(s["gap"], 0)
        x, y = score(reference, candidate, "x"), score(reference, candidate, "y")
        self.assertEqual((x["n"], x["wins"], x["teacher_wins"], y["n"], y["wins"], y["teacher_wins"]), (2, 2, 1, 1, 0, 1))


if __name__ == "__main__":
    unittest.main()
