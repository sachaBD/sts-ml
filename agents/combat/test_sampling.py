"""Invariants of combat training selection probabilities."""
import math
import unittest

from .sampling import deck_distribution


class SamplingInvariants(unittest.TestCase):
    def test_distributions_are_normalized_symmetric_and_concentrate_on_uncertainty(self):
        counts = {str(i): dict(attempts=100, wins=i) for i in range(101)}
        counts['unobserved'] = dict(attempts=0, wins=0)
        for prior in (0.1, 1, 100):
            previous_peak = 0
            for concentration in (0, 0.3, 0.5, 0.9, math.nextafter(1, 0), 1):
                decks = deck_distribution(counts, prior, concentration)
                self.assertAlmostEqual(math.fsum(d['probability'] for d in decks.values()), 1)
                self.assertEqual(set(decks), set(counts))
                for i in range(101):
                    self.assertGreaterEqual(decks[str(i)]['probability'], 0)
                    self.assertAlmostEqual(decks[str(i)]['probability'], decks[str(100-i)]['probability'])
                    self.assertLessEqual(decks[str(i)]['probability'], decks['unobserved']['probability'])
                peak = decks['unobserved']['probability']
                self.assertGreaterEqual(peak, previous_peak)
                previous_peak = peak
                if concentration:
                    self.assertLess(decks['0']['probability'], decks['50']['probability'])
                if concentration == 1:
                    self.assertTrue(all(d['probability'] == 0 for k, d in decks.items()
                                        if k not in ('50', 'unobserved')))
                    self.assertEqual(decks['50']['probability'], decks['unobserved']['probability'])

    def test_half_concentration_is_uncertainty_weighting_and_stronger_priors_flatten_it(self):
        counts = {str(i): dict(attempts=100, wins=i) for i in (0, 25, 50, 75, 100)}
        weighted = deck_distribution(counts, 1, 0.5)
        for d, c in counts.items():
            p = (c['wins'] + 1) / (c['attempts'] + 2)
            self.assertAlmostEqual(weighted[d]['probability'] / weighted['50']['probability'],
                                   p * (1-p) / 0.25)
        stronger = deck_distribution(counts, 100, 0.5)
        self.assertGreater(stronger['0']['probability'], weighted['0']['probability'])
        self.assertLess(stronger['50']['probability'], weighted['50']['probability'])

    def test_equal_evidence_and_zero_concentration_are_uniform(self):
        for counts, concentration in [({str(i): dict(attempts=0, wins=0) for i in range(7)}, 1),
                                      ({str(i): dict(attempts=10, wins=i) for i in range(7)}, 0)]:
            for d in deck_distribution(counts, 1, concentration).values():
                self.assertAlmostEqual(d['probability'], 1 / len(counts))

    def test_impossible_counts_are_rejected(self):
        for attempts, wins in [(-1, 0), (1, -1), (1, 2)]:
            with self.assertRaises(ValueError):
                deck_distribution({'a': dict(attempts=attempts, wins=wins)}, 1, 0.5)


if __name__ == '__main__':
    unittest.main()
