"""A checkpoint must preserve the behavior of continued learning."""
from pathlib import Path
import tempfile
import unittest

import torch

from agents.combat.nn.model import ModelSpec
from agents.combat.nn.training.train import train
from agents.combat.nn.training.learner import Learner


SPEC = ModelSpec('entity', dim=16, layers=1, heads=2, value='win', auxiliary=False)


def train_once(learner, tokens):
    policy = torch.zeros(tokens.shape[:2])
    policy[:, 1:] = 1 / (tokens.shape[1] - 1)  # example_tokens: a context row, then legal actions
    targets = torch.tensor([[1., .5, 0., .1, .2, 3.], [0., 0., .4, .3, 0., 0.]])
    train(learner.net, learner.optimizer, tokens, policy, targets,
          steps=1, batch=2, device='cpu', log=lambda _: None)

class CheckpointTests(unittest.TestCase):
    def test_resuming_from_a_checkpoint_equals_never_stopping(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            learner = Learner(SPEC, lr=1e-3, device='cpu')
            tokens = learner.net.example_tokens(2, 5)
            train_once(learner, tokens)
            checkpoint = Path(temporary) / 'model.pt'
            learner.save(checkpoint)

            train_once(learner, tokens)
            expected = learner.net.eval()(tokens)

            resumed = Learner(SPEC, lr=1e-3, device='cpu', seed=7)
            resumed.load(checkpoint)
            train_once(resumed, tokens)

            torch.testing.assert_close(resumed.net.eval()(tokens), expected, rtol=0, atol=0)


    def test_legacy_checkpoint_preserves_policy_value_and_existing_optimizer_moments(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            source = Learner(SPEC, lr=1e-3, device='cpu')
            tokens = source.net.example_tokens()
            train_once(source, tokens)
            path = Path(temporary) / 'legacy.pt'
            source.save(path)
            checkpoint = torch.load(path, weights_only=True)
            for key in ('thinking.weight', 'thinking.bias'):
                del checkpoint['net'][key]
            group, = checkpoint['optimizer']['param_groups']
            for key in group['params'][-2:]:
                checkpoint['optimizer']['state'].pop(key, None)
            group['params'] = group['params'][:-2]
            torch.save(checkpoint, path)
            restored = Learner(SPEC, lr=1e-3, device='cpu', seed=7)
            restored.load(path)
            torch.testing.assert_close(restored.net.eval()(tokens), source.net.eval()(tokens), rtol=0, atol=0)
            torch.testing.assert_close(restored.optimizer.state_dict()['state'], checkpoint['optimizer']['state'])
            train_once(restored, tokens)


if __name__ == '__main__':
    unittest.main()
