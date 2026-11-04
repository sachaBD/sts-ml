"""Invariants of combat training selection probabilities."""
import math
import unittest

from .sampling import deck_distribution


class SamplingInvariants(unittest.TestCase):
    def test_distributions_are_normalized_with_a_uniform_floor_and_win_loss_symmetry(self):
        counts = {str(i): dict(attempts=100, wins=i) for i in range(101)}
        counts['unobserved'] = dict(attempts=0, wins=0)
        for prior in (0.1, 1, 100):
            for fraction in (0, 0.3, 1):
                decks = deck_distribution(counts, prior, fraction)
                self.assertAlmostEqual(math.fsum(d['probability'] for d in decks.values()), 1)
                for i in range(101):
                    self.assertGreaterEqual(decks[str(i)]['probability'], (1 - fraction) / len(counts))
                    self.assertAlmostEqual(decks[str(i)]['probability'], decks[str(100-i)]['probability'])
                    self.assertLessEqual(decks[str(i)]['probability'], decks['unobserved']['probability'])
                self.assertEqual(set(decks), set(counts))
                if fraction:
                    self.assertLess(decks['0']['probability'], decks['50']['probability'])

    def test_mixture_is_linear_and_stronger_priors_flatten_extreme_outcomes(self):
        counts = {str(i): dict(attempts=100, wins=i) for i in (0, 25, 50, 75, 100)}
        uniform = deck_distribution(counts, 1, 0)
        weighted = deck_distribution(counts, 1, 1)
        mixed = deck_distribution(counts, 1, 0.4)
        for d in counts:
            self.assertAlmostEqual(mixed[d]['probability'],
                                   0.6 * uniform[d]['probability'] + 0.4 * weighted[d]['probability'])
        stronger = deck_distribution(counts, 100, 1)
        self.assertGreater(stronger['0']['probability'], weighted['0']['probability'])
        self.assertLess(stronger['50']['probability'], weighted['50']['probability'])

    def test_equal_evidence_and_zero_fraction_are_uniform(self):
        for counts, fraction in [({str(i): dict(attempts=0, wins=0) for i in range(7)}, 1),
                                 ({str(i): dict(attempts=10, wins=i) for i in range(7)}, 0)]:
            for d in deck_distribution(counts, 1, fraction).values():
                self.assertAlmostEqual(d['probability'], 1 / len(counts))

    def test_impossible_counts_are_rejected(self):
        for attempts, wins in [(-1, 0), (1, -1), (1, 2)]:
            with self.assertRaises(ValueError):
                deck_distribution({'a': dict(attempts=attempts, wins=wins)}, 1, 0.5)


if __name__ == '__main__':
    unittest.main()
