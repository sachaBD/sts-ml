"""Boss selection for act 1 bootstrap runs."""
import unittest

from apps.bootstrap.generate import ACT1_BOSSES, selected_bosses, simulation_budgets, teacher_options


class BossSelectionTests(unittest.TestCase):
    def test_required_and_subset(self):
        with self.assertRaises(SystemExit):
            selected_bosses({})
        self.assertEqual(selected_bosses({"bosses": list(ACT1_BOSSES)}), list(ACT1_BOSSES))
        self.assertEqual(selected_bosses({"bosses": ["hexaghost", "slime_boss"]}),
                         ["hexaghost", "slime_boss"])

    def test_teacher_options(self):
        self.assertEqual(teacher_options({}), {})
        self.assertEqual(teacher_options({"stop_factor": 0.5, "merge_identical_cards": True}),
                         {"stop_factor": 0.5, "merge_identical_cards": True})
        for options in ({"stop_factor": 0}, {"stop_factor": 2}, {"stop_factor": True},
                        {"merge_identical_cards": 1}):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                teacher_options(options)

    def test_simulation_budgets(self):
        full = {"easy": 500, "hard": 2000, "elite": 5000, "event": 5000, "boss": 15000}
        self.assertEqual(simulation_budgets({"simulations": full}), full)
        with self.assertRaises(SystemExit):
            simulation_budgets({})
        for budgets in ([], {"easy": 500, "boss": 15000}, {**full, "unknown": 100}, {**full, "easy": 0},
                        {**full, "boss": True}, {**full, "event": 2.5}):
            with self.subTest(budgets=budgets), self.assertRaises(SystemExit):
                simulation_budgets({"simulations": budgets})

    def test_invalid_selection(self):
        for bosses in ([], "slime_boss", ["unknown"], ["slime_boss", "slime_boss"], [1]):
            with self.subTest(bosses=bosses), self.assertRaises(SystemExit):
                selected_bosses({"bosses": bosses})


if __name__ == "__main__":
    unittest.main()
