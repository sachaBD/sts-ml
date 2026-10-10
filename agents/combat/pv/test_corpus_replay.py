import unittest

from apps.combat_expert_iteration.corpus_manifest import parse_manifest
from apps.combat_expert_iteration.test_corpus_manifest import manifest
from .corpus_replay import Fight, SLOTS_PER_EPOCH, STATES_PER_SLOT, plan_epoch


class ReplayPlanTests(unittest.TestCase):
    def setUp(self):
        self.manifest = parse_manifest(manifest())

    def fights_for_round(self, round_number):
        out = []
        for family in self.manifest.families:
            if family.role == "original":
                out += [Fight(f"{family.family_id}-t", family.family_id, "teacher", 0, 3),
                        Fight(f"{family.family_id}-l", family.family_id, "learner", 0, 5)]
            elif family.role == "pilot" and family.wave <= round_number:
                out.append(Fight(f"{family.family_id}-t", family.family_id, "teacher", family.wave, 3))
                out.append(Fight(f"{family.family_id}-l", family.family_id, "learner", family.wave, 5))
                if family.wave < round_number:  # this round's revisit learner collection
                    out.append(Fight(f"{family.family_id}-revisit-{round_number}", family.family_id, "learner", round_number, 7))
        return out

    def test_exact_family_first_slots_states_and_reproducibility(self):
        fights = self.fights_for_round(2)
        first = plan_epoch(self.manifest.families, fights, 2, 71)
        second = plan_epoch(tuple(reversed(self.manifest.families)), list(reversed(fights)), 2, 71)
        self.assertEqual(first, second)
        self.assertNotEqual(first, plan_epoch(self.manifest.families, fights, 2, 72))
        self.assertEqual(len(first.slots), SLOTS_PER_EPOCH)
        self.assertEqual(first.state_count, 192000)
        counts = first.counts()
        for family in [f"old{i}" for i in range(10)]:
            self.assertEqual((counts[family, "teacher"], counts[family, "historical_learner"]), (20, 80))
        for family in [f"w2-{i}" for i in range(10)]:
            self.assertEqual((counts[family, "teacher"], counts[family, "current_learner"]), (20, 80))
        for family in [f"w1-{i}" for i in range(10)]:
            self.assertEqual((counts[family, "teacher"], counts[family, "current_learner"], counts[family, "historical_learner"]), (20, 40, 40))
        self.assertTrue(all(len(slot.state_rows) == STATES_PER_SLOT for slot in first.slots))
        row_counts = {fight.fight_id: fight.row_count for fight in fights}
        self.assertTrue(all(0 <= row < row_counts[slot.fight_id]
                            for slot in first.slots for row in slot.state_rows))
        mixed = first.mixed_state_draws()
        self.assertEqual(len(mixed), 192000)
        self.assertGreater(len({x[2] for x in mixed[:64]}), 1)  # never batch raw per-slot state order

    def test_all_rounds_have_exact_family_allocation(self):
        for round_number in range(1, 5):
            plan = plan_epoch(self.manifest.families, self.fights_for_round(round_number), round_number, 19)
            per_family = SLOTS_PER_EPOCH // (10 + 10 * round_number)
            self.assertEqual(len(plan.slots), SLOTS_PER_EPOCH)
            self.assertEqual({sum(1 for slot in plan.slots if slot.family_id == family.family_id)
                              for family in self.manifest.families if family.role == "original" or
                              (family.role == "pilot" and family.wave <= round_number)}, {per_family})

    def test_fallback_and_missing_pool_fail_closed(self):
        fights = self.fights_for_round(1)
        # Wave one has no history, so its current learner receives the full 80%.
        plan = plan_epoch(self.manifest.families, fights, 1, 1)
        self.assertEqual(plan.counts()["w1-0", "current_learner"], 120)
        missing_teacher = [x for x in fights if x.fight_id != "old0-t"]
        with self.assertRaisesRegex(ValueError, "missing teacher"):
            plan_epoch(self.manifest.families, missing_teacher, 1, 1)
        missing_learners = [x for x in fights if x.family_id != "old0" or x.source != "learner"]
        with self.assertRaisesRegex(ValueError, "missing both learner"):
            plan_epoch(self.manifest.families, missing_learners, 1, 1)

    def test_rejects_future_heldout_original_fresh_and_duplicate(self):
        fights = self.fights_for_round(1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            plan_epoch(self.manifest.families, fights + [fights[0]], 1, 1)
        with self.assertRaisesRegex(ValueError, "held-out"):
            plan_epoch(self.manifest.families, fights + [Fight("dev", "dev", "teacher", 0, 1)], 1, 1)
        with self.assertRaisesRegex(ValueError, "replay-only"):
            plan_epoch(self.manifest.families, fights + [Fight("fresh-old", "old0", "learner", 1, 1)], 1, 1)
        with self.assertRaisesRegex(ValueError, "future"):
            plan_epoch(self.manifest.families, fights + [Fight("future", "w1-0", "teacher", 2, 1)], 1, 1)
        with self.assertRaisesRegex(ValueError, "integer"):
            plan_epoch(self.manifest.families, fights + [Fight("bad", "old0", "teacher", 0, True)], 1, 1)
        with self.assertRaisesRegex(ValueError, "round_number"):
            plan_epoch(self.manifest.families, fights, True, 1)


if __name__ == "__main__":
    unittest.main()
