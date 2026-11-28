import json
import unittest
from unittest.mock import patch

from . import session


class SessionTests(unittest.TestCase):
    def test_counts_match_calls_and_pairs_share_a_world(self):
        with patch.object(session, 'Q', wraps=session.Q) as q, patch.object(session, 'S', wraps=session.S) as sim:
            s = session.Session(seed=4, budget=2, price=0)
            world = s.game.sample()
            self.assertEqual(q.call_count, 1)
            self.assertEqual(sim.call_count, 0)
            with patch.object(s.game, 'sample', return_value=world):
                selected = s.decision.select(s.price, s.budget)
                row = next(r for r in s.candidates() if r['id'] == selected.name)
                observation = s.step()
                calls = s.snapshot()
                a, b = observation['branches']
                self.assertEqual(a['path'][0]['state'], world.world.intent)
                self.assertEqual(b['path'][0]['state'], world.world.intent)
                self.assertEqual((a['move'], b['move']), (row['left'], row['right']))
                self.assertEqual(calls['s_calls'], len(a['path']) + len(b['path']))
                self.assertEqual(calls['q_calls'], q.call_count)
                self.assertEqual(calls['s_calls'], sim.call_count)
                self.assertEqual(observation['observation'], a['value'] - b['value'])
                # Replaying the same paths hits the cache but still simulates each action.
                for branch in (a, b):
                    s.rollout(world, branch['move'])
                self.assertEqual(s.snapshot()['q_calls'], calls['q_calls'])
                self.assertEqual(s.snapshot()['s_calls'], 2*calls['s_calls'])
            self.assertTrue(all(call.args[0].phase not in ('Resolve', 'Terminal') for call in q.call_args_list))

    def test_stop_preserves_counts_and_prior(self):
        for s in (session.Session(price=1.), session.Session(budget=0)):
            before = s.snapshot()
            self.assertIsNone(s.step())
            self.assertEqual(s.snapshot(), before)
            self.assertEqual(s.grade()['no_search'], s.grade()['search'])

    def test_candidates_never_sample_worlds(self):
        s = session.Session()
        with patch.object(s.game, 'sample', side_effect=AssertionError):
            rows = s.candidates()
        moves = s.snapshot()['actions']
        pairs = {frozenset((row['left'], row['right'])) for row in rows}
        self.assertEqual(pairs, {frozenset((a, b)) for a in moves for b in moves if a != b})
        self.assertEqual(len(rows), len(pairs))
        self.assertTrue(all(row['cost'] == 1 for row in rows))
        json.dumps([s.snapshot(), rows], allow_nan=False)
