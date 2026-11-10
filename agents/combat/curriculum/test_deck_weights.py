"""Invariants of the deck weights."""
import math
import unittest

from agents.combat.curriculum.deck_weights import deck_weights


class DeckDistributionTests(unittest.TestCase):
    def test_distributions_are_normalized_symmetric_and_concentrate_on_uncertainty(self):
        counts = {str(i): dict(attempts=100, wins=i) for i in range(101)}
        counts['unobserved'] = dict(attempts=0, wins=0)
        for prior in (0.1, 1, 100):
            previous_peak = 0
            for concentration in (0, 0.3, 0.5, 0.9, math.nextafter(1, 0), 1):
                decks = deck_weights(counts, prior, concentration)
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

    def test_impossible_counts_are_rejected(self):
        for attempts, wins in [(-1, 0), (1, -1), (1, 2)]:
            with self.assertRaises(ValueError):
                deck_weights({'a': dict(attempts=attempts, wins=wins)}, 1, 0.5)


if __name__ == '__main__':
    unittest.main()
