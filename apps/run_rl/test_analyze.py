import copy
import unittest

from apps.run_rl.analyze import mean_se, paired, summary


class AnalyzeTests(unittest.TestCase):
    def test_synthetic_runs(self):
        run = {"seed": 7, "status": "died", "floor": 20, "act": 2, "acts_cleared": 1,
               "steps": [
                   {"kind": "fight", "category": "elite", "encounter": "nob", "won": True,
                    "hp_before": 60, "state": {"act": 1}},
                   {"kind": "fight", "category": "boss", "encounter": "guardian", "won": True,
                    "hp_before": 40, "state": {"act": 1}},
                   {"kind": "decide", "decision": "boss_relic", "choice": 0, "source": "net",
                    "options": [{"relic": "sozu"}, {"relic": "skip"}]},
                   {"kind": "fight", "category": "easy", "encounter": "thieves", "won": False,
                    "hp_before": 70, "state": {"act": 2}}]}
        other = copy.deepcopy(run)
        other.update(seed=8, status="act_complete", floor=33, acts_cleared=2)
        other["steps"][-1]["won"] = True
        other["steps"][2].update(choice=1, source="explore")
        text = summary([run, other], "synthetic")
        for expected in ["n: 2", "1: 1, 2: 1", "Deaths by act: 2: 1", "Elites/run (elites: runs): 1: 2",
                         "| easy | thieves | 1 | 2 | 50.0% |", "| sozu | 1 | 1 | 0 | 0 |",
                         "skip rate: 50.0% (1/2)", "40.000 ± 0.000 (n=2)", "70.000 ± 0.000 (n=2)"]:
            self.assertIn(expected, text)
        matched = copy.deepcopy(other)
        matched["seed"] = 7
        diffs = paired([run], [matched])
        self.assertIn("Act 1 clear mean difference ± SE: 0.000 ± n/a (n=1)", diffs)
        self.assertIn("Act 2 clear mean difference ± SE: 1.000 ± n/a (n=1)", diffs)
        self.assertIn("n/a (n=0)", mean_se([]))
        self.assertIn("3.000 ± 1.000 (n=2)", mean_se([2, 4]))


if __name__ == "__main__":
    unittest.main()
