"""Invariants of adaptive sampling: causal history windows and repeatable recovery."""
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from .config import load
from .deck_stats import iteration_counts
from .sampling import deck_distribution, sampling_weights
from .test_expert_iteration import CONFIG


class SamplingInvariants(unittest.TestCase):
    def test_windows_are_additive_exclude_future_data_and_recovery_reuses_saved_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            training = Path(tmp)
            rng = random.Random(71)
            counts = {d: dict(fights=0, wins=0) for d in 'abc'}
            for it in range(8):
                directory = training / f'iter{it:03d}'
                directory.mkdir()
                for c in counts.values():
                    attempts = rng.randrange(1, 20) if it else 0
                    c['fights'] += attempts
                    c['wins'] += rng.randrange(attempts + 1)
                (directory / 'decks.json').write_text(json.dumps(counts))
            whole = iteration_counts(training, 6, 6)
            first, last = iteration_counts(training, 3, 3), iteration_counts(training, 6, 3)
            for d in counts:
                for key in ('attempts', 'wins'):
                    self.assertEqual(whole[d][key], first[d][key] + last[d][key])
            settings = load(CONFIG.replace('per_iteration = 5',
                'per_iteration = 5\nhistory_iterations = 3\nsampling_concentration = 0.5')).starts
            weights = sampling_weights(training, 7, settings)
            artifact = json.loads((training / 'sampling/iter007.json').read_text())
            for d in counts:
                self.assertEqual(artifact['decks'][d]['attempts'], last[d]['attempts'])
                self.assertEqual(artifact['decks'][d]['wins'], last[d]['wins'])
            # Recovery archives the interrupted iteration; it must neither read future results nor re-estimate.
            (training / 'iter007').rename(training / '.interrupted-iter007')
            with patch('apps.gumbel_nn_combat_expert_iteration.sampling.iteration_counts', side_effect=AssertionError):
                self.assertEqual(sampling_weights(training, 7, settings), weights)


class DeckDistributionTests(unittest.TestCase):
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
