"""Invariants of adaptive sampling: causal history windows and repeatable recovery."""
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from .config import load
from .sampling import iteration_counts, sampling_weights
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


if __name__ == '__main__':
    unittest.main()
