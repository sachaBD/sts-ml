"""A checkpoint must preserve the behavior of continued learning."""
from pathlib import Path
import tempfile
import unittest

import torch

from ._test_support import combat_tokens, small_spec, train_once
from .learner import Learner


class CheckpointTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_learning(self):
        torch.set_num_threads(1)
        for kind in ('transformer', 'entity'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                learner = Learner(small_spec(kind), lr=1e-3, device='cpu')
                tokens = combat_tokens()
                train_once(learner, tokens)
                checkpoint = Path(temporary) / 'model.pt'
                learner.save(checkpoint)

                train_once(learner, tokens)
                expected = learner.net.eval()(tokens)

                resumed = Learner(small_spec(kind), lr=1e-3, device='cpu', seed=7)
                resumed.load(checkpoint)
                train_once(resumed, tokens)

                torch.testing.assert_close(resumed.net.eval()(tokens), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
