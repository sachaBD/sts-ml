import unittest

import numpy as np

from .model import Belief, Chance, Choice, Leaf, Measurement, adaptive_belief, envelope_gain, policies


class ModelTests(unittest.TestCase):
    def test_can_adapt_only_after_intent_is_observed(self):
        def pick(i):
            return Choice({'Strike': Leaf(i), 'Defend': Leaf(1-i)})
        after = Chance({'hit': (.5, pick(0)), 'rest': (.5, pick(1))})
        before = Choice({a: Chance({i: (p, c.actions[a]) for i,(p,c) in after.outcomes.items()})
                         for a in pick(0).actions})
        def best(node):
            return max(w @ [0., 1.] for _, w in policies(node, 2))
        self.assertEqual(best(after), 1.)
        self.assertEqual(best(before), .5)

    def test_no_value_from_common_offset_or_identical_plans(self):
        self.assertEqual(envelope_gain([.8,.3,.5], [2.,2.,2.]), 0.)
        b = Belief([.5,.5], np.eye(2))
        m = Measurement('one leaf', np.array([1.,0.]), 1., 1)
        self.assertEqual(b.voi(m, np.array([[1.,0.],[1.,0.]])), 0.)

    def test_paired_probe_does_not_update_common_mode(self):
        b = Belief([.5,.5], np.eye(2))
        m = Measurement('difference', np.array([1.,-1.]), .5, 2)
        self.assertAlmostEqual(b.prediction(m)[0], 0.)
        b.observe(m, 1.)
        np.testing.assert_allclose(b.mean, [.9,.1])
        np.testing.assert_allclose(b.cov @ [1.,1.], [1.,1.])
        self.assertLess(m.h @ b.cov @ m.h, 2.)

    def test_upper_envelope_matches_sampled_reoptimization(self):
        rng = np.random.default_rng(123)
        reaches = np.array([[1,0,0], [0,.8,.2], [0,.4,.6], [0,0,1.]])
        b = adaptive_belief([.55,.4,.45], [[.02,.01,0],[.01,.03,0],[0,0,.02]])
        m = Measurement('compare', np.array([1.,-1.,0.]), .01, 2)
        # Simulate latent returns then noisy probes, conditional on the regime.
        s = rng.choice(b.scales, size=100000, p=np.exp(b.log_weights))
        x = b.mean + s[:,None]*rng.multivariate_normal(np.zeros(3), b.cov, size=len(s))
        y = x @ m.h + s*rng.normal(0, m.noise_var**.5, size=len(s))
        kalman_gain = b.cov @ m.h / (m.h @ b.cov @ m.h + m.noise_var)
        updated = b.mean + (y - b.mean @ m.h)[:,None]*kalman_gain
        gains = np.max(updated @ reaches.T, axis=1) - max(reaches @ b.mean)
        self.assertAlmostEqual(b.voi(m, reaches), gains.mean(), delta=5*gains.std()/len(s)**.5)

    def test_surprising_jump_can_warrant_rechecking_new_leader(self):
        adaptive = adaptive_belief([.35,.5], np.diag([.01,0.]))
        fixed = Belief(adaptive.mean, adaptive.cov, scales=(1.8**.5,), probabilities=(1.,))
        m = Measurement('challenger', np.array([1.,0.]), .01, 1)
        # Same prior predictive variance and same posterior means. Only the
        # adaptive model treats the jump as evidence about future uncertainty.
        for b in [adaptive, fixed]:
            b.observe(m, .915)
            self.assertEqual(b.mean.argmax(), 0)
        np.testing.assert_allclose(adaptive.mean, fixed.mean)
        price = .003
        self.assertLess(fixed.voi(m, np.eye(2)), price)
        self.assertGreater(adaptive.voi(m, np.eye(2)), price)
        self.assertGreater(np.exp(adaptive.log_weights[1]), .9)


if __name__ == '__main__':
    unittest.main()
