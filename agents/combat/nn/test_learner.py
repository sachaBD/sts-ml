"""Checkpoints preserve training continuation, not just predictions."""
from pathlib import Path
import tempfile
import unittest

import torch

from .learner import Learner
from .model import ModelSpec
from .test_model import tokens


def step(learner, x):
    learner.optimizer.zero_grad()
    value, logits = learner.net(x)
    loss = -value.log().mean() - logits[:, 7:10].log_softmax(-1).mean()
    loss.backward()
    learner.optimizer.step()


class LearnerTests(unittest.TestCase):
    def test_resume_preserves_optimizer_rng_and_next_update(self):
        torch.set_num_threads(1)
        for kind in ('transformer', 'entity'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                learner = Learner(ModelSpec(kind, 16, 1, 2), 1e-3, 'cpu')
                x = tokens()
                step(learner, x)
                path = Path(tmp) / 'model.pt'
                learner.save(path)
                expected_rng = torch.rand(8)
                step(learner, x)
                resumed = Learner(learner.net.spec, 1e-3, 'cpu', seed=5)
                resumed.load(path)
                torch.testing.assert_close(torch.rand(8), expected_rng, rtol=0, atol=0)
                step(resumed, x)
                for name, weight in learner.net.state_dict().items():
                    torch.testing.assert_close(weight, resumed.net.state_dict()[name], rtol=0, atol=0)
                other = 'entity' if kind == 'transformer' else 'transformer'
                incompatible = Learner(ModelSpec(other, 16, 1, 2), 1e-3, 'cpu')
                with self.assertRaisesRegex(ValueError, 'architecture'):
                    incompatible.load(path)


if __name__ == '__main__':
    unittest.main()
