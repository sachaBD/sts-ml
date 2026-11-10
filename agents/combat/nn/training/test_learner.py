"""A checkpoint must preserve the behavior of continued learning."""
from pathlib import Path
import tempfile
import unittest

import torch

from agents.combat.nn.entity.test_model import combat_tokens, small_spec
from agents.combat.nn.tokens import TokenKind, TokenRows
from agents.combat.nn.training.train import train
from agents.combat.nn.training.learner import Learner


def train_once(learner, tokens):
    actions = TokenRows(tokens).is_kind(TokenKind.ACTION).float()
    policy = actions / actions.sum(dim=1, keepdim=True)
    # won, hp, enemy_hp, hp_lost, enemy_hp_lost, turns_remaining
    targets = torch.tensor([[1., .5, 0., .1, .2, 3.], [0., 0., .4, .3, 0., 0.]])
    train(learner.net, learner.optimizer, tokens, policy, targets,
          steps=1, batch=2, device='cpu', log=lambda _: None)

class CheckpointTests(unittest.TestCase):
    def test_resuming_from_a_checkpoint_equals_never_stopping(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            learner = Learner(small_spec(), lr=1e-3, device='cpu')
            tokens = combat_tokens()
            train_once(learner, tokens)
            checkpoint = Path(temporary) / 'model.pt'
            learner.save(checkpoint)

            train_once(learner, tokens)
            expected = learner.net.eval()(tokens)

            resumed = Learner(small_spec(), lr=1e-3, device='cpu', seed=7)
            resumed.load(checkpoint)
            train_once(resumed, tokens)

            torch.testing.assert_close(resumed.net.eval()(tokens), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
