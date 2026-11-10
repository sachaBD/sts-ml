"""A checkpoint must preserve the behavior of continued learning."""
from pathlib import Path
import tempfile
import unittest

import torch

from agents.combat.nn.entity.test_model import combat_tokens, small_spec
from agents.combat.nn.tokens import TokenKind, TokenRows
from agents.combat.nn.training.train import train
from agents.combat.nn.training.learner import Learner
from agents.combat.nn.training.replay import Replay
from agents.combat.nn.training.test_replay import samples



def train_once(learner, tokens):
    actions = TokenRows(tokens).is_kind(TokenKind.ACTION).float()
    policy = actions / actions.sum(dim=1, keepdim=True)
    # won, hp, enemy_hp, hp_lost, enemy_hp_lost, turns_remaining
    targets = torch.tensor([[1., .5, 0., .1, .2, 3.], [0., 0., .4, .3, 0., 0.]])
    train(learner.net, learner.optimizer, tokens, policy, targets,
          steps=1, batch=2, device='cpu', log=lambda _: None)

class CheckpointTests(unittest.TestCase):
    def test_sparse_supervision_checkpoint_restores_both_sampling_streams(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = samples(root / 'data', [('a', 7, 1, 1), ('a', 3, 2, 0), ('b', 5, 3, 1)])
            learner = Learner(small_spec(), lr=1e-3, device='cpu')
            with Replay([path], balanced=False) as replay:
                learner.fit(replay, 2, 5, value_batch=2, log=lambda _: None)
                learner.save(root / 'model.pt')
                learner.fit(replay, 3, 5, value_batch=2, log=lambda _: None)
            expected_rng = torch.get_rng_state()
            resumed = Learner(small_spec(), lr=1e-3, device='cpu', seed=7)
            resumed.load(root / 'model.pt')
            with Replay([path], balanced=False) as replay:
                resumed.fit(replay, 3, 5, value_batch=2, log=lambda _: None)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
            for name, parameter in learner.net.state_dict().items():
                torch.testing.assert_close(resumed.net.state_dict()[name], parameter, rtol=0, atol=0)

    def test_a_given_learning_rate_replaces_the_checkpoint_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = samples(root / 'data', [('a', 7, 1, 1), ('b', 5, 3, 1)])
            Learner(small_spec(), lr=1e-3, device='cpu').save(root / 'model.pt')
            warm = Learner(small_spec(), lr=1e-3, device='cpu')
            warm.load(root / 'model.pt')
            with Replay([path], balanced=False) as replay:
                warm.fit(replay, 1, 5, log=lambda _: None, lr=1e-4)
            self.assertEqual({group['lr'] for group in warm.optimizer.param_groups}, {1e-4})

    def test_resume_matches_uninterrupted_learning(self):
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
