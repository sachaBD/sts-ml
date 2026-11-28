import json
import unittest

from .session import Session


class SessionTests(unittest.TestCase):
    def test_budget_manual_probe_and_seeded_replay(self):
        s = Session(seed=5, budget=3)
        first = s.step()['id']
        second = next(row['id'] for row in s.candidates() if row['cost'] == 1)
        s.step(second)
        replay = Session(seed=5, budget=3)
        for key in (first, second):
            replay.step(key)
        self.assertEqual(replay.snapshot(), s.snapshot())
        with self.assertRaises(ValueError):
            s.step(next(r['id'] for r in s.candidates() if r['cost'] == 2))
        json.dumps([s.snapshot(), s.candidates()], allow_nan=False)

    def test_stop_preserves_prior_when_too_expensive_or_unaffordable(self):
        for s in (Session(price=1.), Session(budget=0)):
            self.assertIsNone(s.step())
            self.assertEqual(s.grade()['no_search'], s.grade()['search'])
            self.assertEqual(s.spent, 0)

    def test_ranking_never_reads_hidden_truth(self):
        s = Session()
        rows = s.candidates()
        s.game.truth[:] = float('nan')
        self.assertEqual(rows, s.candidates())


if __name__ == '__main__':
    unittest.main()
