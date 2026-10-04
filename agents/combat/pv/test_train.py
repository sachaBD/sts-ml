"""Focused win-contract value-target checks; no training or play jobs."""
import unittest

import torch

from .model import NAMES
from .train import evaluate


class ValueMix(unittest.TestCase):
    def test_half_mix_keeps_no_root_outcome(self):
        batch = ({name: torch.zeros(4, 1) for name in NAMES},
                 torch.tensor([100., 0., 100., 0.]), torch.ones(4, 1), torch.ones(4, dtype=torch.bool),
                 torch.tensor([20., 60., float('nan'), 0.]))
        # Expected targets: 60, 30, unchanged 100 (forced/no root), and 0 (valid zero-valued root).
        net = lambda *inputs: (torch.tensor([60., 30., 100., 0.]), torch.zeros(4, 1))
        loss, *_ = evaluate(net, batch, 'cpu', value_mix=0.5)
        self.assertEqual(loss.item(), 0)
        for bad in (-1., 101., float('inf')):
            with self.subTest(root=bad), self.assertRaisesRegex(ValueError, r'\[0, 100\]'):
                evaluate(net, (*batch[:4], torch.tensor([bad, 60., float('nan'), 0.])), 'cpu', value_mix=0.5)
        with self.assertRaisesRegex(ValueError, 'old HP units'):
            evaluate(net, batch, 'cpu', teacher_root_mix=0.5)


if __name__ == '__main__':
    unittest.main()
