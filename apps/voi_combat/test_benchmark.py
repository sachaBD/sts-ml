import unittest

import numpy as np

from .benchmark import CombatDecision


class CombatAdapterTests(unittest.TestCase):
    def test_zero_budget_preserves_policy_including_ties(self):
        d = CombatDecision({'value': .5, 'logits': [-2, 3, 3, 1, 2]})
        self.assertIsNone(d.select(0, 0))
        self.assertEqual(d.move(), 1)

    def test_native_pair_matches_signed_linear_measurement(self):
        logits = np.array([-1., .5, 1., 2., -4.])
        d = CombatDecision({'value': .5, 'logits': logits})
        m = d.select(0., 1)
        a,b = d.pair(m)
        self.assertEqual(logits.argmax(), b)
        self.assertEqual(logits[a]-logits[b], m.h @ logits[d.actions])
        d.belief.observe(m, 1.)
        self.assertEqual(d.move(), a)


if __name__ == '__main__':
    unittest.main()
