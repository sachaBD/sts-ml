"""Checkpoint guarantees are measured through predictions and training continuation."""
from pathlib import Path
import tempfile
import unittest

import torch

from ._test_support import combat_tokens, small_spec, train_once
from .learner import Learner
from .model import ModelSpec


class CheckpointContract:
    def setUp(self):
        torch.set_num_threads(1)
        self.learner = Learner(small_spec(self.kind), lr=1e-3, device='cpu')
        self.tokens = combat_tokens()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.checkpoint = Path(temporary.name) / 'model.pt'

    def restore(self):
        restored = Learner(small_spec(self.kind), lr=1e-3, device='cpu', seed=7)
        restored.load(self.checkpoint)
        return restored

    def test_training_can_change_predictions(self):
        before_values, before_logits = self.learner.net.eval()(self.tokens)

        train_once(self.learner, self.tokens)

        after_values, after_logits = self.learner.net.eval()(self.tokens)
        unchanged = torch.equal(before_values, after_values) and torch.equal(before_logits, after_logits)
        self.assertFalse(unchanged)

    def test_checkpoint_restores_predictions(self):
        train_once(self.learner, self.tokens)
        expected = self.learner.net.eval()(self.tokens)
        self.learner.save(self.checkpoint)

        restored = self.restore()

        torch.testing.assert_close(restored.net.eval()(self.tokens), expected, rtol=0, atol=0)

    def test_checkpoint_restores_random_sequence(self):
        self.learner.save(self.checkpoint)
        expected = torch.rand(8)

        self.restore()

        torch.testing.assert_close(torch.rand(8), expected, rtol=0, atol=0)

    def test_resume_reproduces_the_next_training_update(self):
        train_once(self.learner, self.tokens)
        self.learner.save(self.checkpoint)
        train_once(self.learner, self.tokens)
        expected = self.learner.net.eval()(self.tokens)

        resumed = self.restore()
        train_once(resumed, self.tokens)

        torch.testing.assert_close(resumed.net.eval()(self.tokens), expected, rtol=0, atol=0)

    def test_different_architecture_cannot_load_checkpoint(self):
        self.learner.save(self.checkpoint)
        different = ModelSpec(self.kind, dim=32, layers=1, heads=2)
        incompatible = Learner(different, lr=1e-3, device='cpu')

        with self.assertRaises(ValueError):
            incompatible.load(self.checkpoint)


class TransformerCheckpointTests(CheckpointContract, unittest.TestCase):
    kind = 'transformer'


class PooledTransformerCheckpointTests(CheckpointContract, unittest.TestCase):
    kind = 'entity'


if __name__ == '__main__':
    unittest.main()
