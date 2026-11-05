"""A checkpoint must preserve the behavior of continued learning."""
from pathlib import Path
import tempfile
import unittest

import torch

from ._test_support import combat_tokens, small_spec, train_once
from .learner import Learner
from .replay import Replay
from .test_replay import samples
from .tokens import TokenKind, TokenRows
from .train import losses


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


@unittest.skipUnless(torch.cuda.is_available(), 'compiled training runs on CUDA')
class CompileTests(unittest.TestCase):
    def test_compile_changes_no_learner_state_and_preserves_losses(self):
        tokens = combat_tokens()
        actions = TokenRows(tokens).is_kind(TokenKind.ACTION).float()
        policy, outcomes = actions / actions.sum(dim=1, keepdim=True), torch.tensor([1., 0.])
        for kind in ('transformer', 'entity'):
            with self.subTest(kind=kind):
                learner = Learner(small_spec(kind), lr=1e-3, device='cuda')
                weights = {name: p.clone() for name, p in learner.net.state_dict().items()}
                rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                learner.compile()
                self.assertEqual(learner.net.state_dict().keys(), weights.keys())
                for name, parameter in learner.net.state_dict().items():
                    torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in learner.net.parameters()))
                self.assertEqual(learner.optimizer.state_dict()['state'], {})
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for after, before in zip(torch.cuda.get_rng_state_all(), cuda_rng):
                    torch.testing.assert_close(after, before, rtol=0, atol=0)
                learner.net.train()
                compiled = torch.stack(losses(learner.training_net, tokens, policy, outcomes, 'cuda'))
                eager = torch.stack(losses(learner.net, tokens, policy, outcomes, 'cuda'))
                torch.testing.assert_close(compiled, eager, rtol=1e-2, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
