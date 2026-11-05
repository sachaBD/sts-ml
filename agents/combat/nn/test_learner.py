"""A checkpoint must preserve the behavior of continued learning."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from ._test_support import combat_tokens, small_spec, train_once
from .learner import Learner
from .replay import Replay
from .test_replay import samples


class CheckpointTests(unittest.TestCase):
    def test_sparse_supervision_checkpoint_restores_both_sampling_streams(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = samples(root / 'data', [('a', 7, 1, 1), ('a', 3, 2, 0), ('b', 5, 3, 1)])
            learner = Learner(small_spec('entity'), lr=1e-3, device='cpu')
            with Replay([path], balanced=False) as replay:
                learner.fit(replay, 2, 5, value_batch=2, log=lambda _: None)
                learner.save(root / 'model.pt')
                learner.fit(replay, 3, 5, value_batch=2, log=lambda _: None)
            expected_rng = torch.get_rng_state()
            resumed = Learner(small_spec('entity'), lr=1e-3, device='cpu', seed=7)
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
            Learner(small_spec('entity'), lr=1e-3, device='cpu').save(root / 'model.pt')
            warm = Learner(small_spec('entity'), lr=1e-3, device='cpu')
            warm.load(root / 'model.pt')
            with Replay([path], balanced=False) as replay:
                warm.fit(replay, 1, 5, log=lambda _: None, lr=1e-4)
            self.assertEqual({group['lr'] for group in warm.optimizer.param_groups}, {1e-4})

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

    def test_target_association_is_an_explicit_checkpoint_semantic(self):
        for kind in ('transformer', 'entity'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                checkpoint = Path(temporary) / 'model.pt'
                spec = small_spec(kind)
                learner = Learner(spec, lr=1e-3, device='cpu')
                tokens = combat_tokens()
                tokens[:, 7:, -1] = 0  # legacy native encoder's reserved column
                expected = learner.net.eval()(tokens)
                learner.save(checkpoint)
                saved = torch.load(checkpoint, weights_only=True)
                del saved['spec']['target_association']
                torch.save(saved, checkpoint)
                resumed = Learner(spec, lr=1e-3, device='cpu')
                resumed.load(checkpoint)
                torch.testing.assert_close(resumed.net.eval()(tokens), expected, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'architecture'):
                    Learner(replace(spec, target_association=True), lr=1e-3, device='cpu').load(checkpoint)


if __name__ == '__main__':
    unittest.main()
