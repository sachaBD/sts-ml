"""Focused win-contract value-target checks; no training or play jobs."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .model import NAMES
from .train import evaluate, run_epoch, main


class ValueMix(unittest.TestCase):
    def test_half_mix_keeps_no_root_outcome(self):
        batch = ({name: torch.zeros(4, 1) for name in NAMES},
                 torch.tensor([100., 0., 100., 0.]), torch.ones(4, 1), torch.ones(4, dtype=torch.bool),
                 torch.tensor([20., 60., float('nan'), 0.]))
        # Expected targets: 60, 30, unchanged 100 (forced/no root), and 0 (valid zero-valued root).
        net = lambda *inputs: (torch.tensor([60., 30., 100., 0.]), torch.zeros(4, 1))
        loss, *_ = evaluate(net, batch, 'cpu', value_mix=0.5)
        self.assertEqual(loss.item(), 0)
        clipped_net = lambda *inputs: (torch.tensor([100., 30., 100., 0.]), torch.zeros(4, 1))
        loss, *_ = evaluate(clipped_net, (*batch[:4], torch.tensor([121.8, 60., float('nan'), 0.])),
                            'cpu', value_mix=0.5)
        self.assertEqual(loss.item(), 0)  # 121.8 is clamped to 100 before mixing with outcome 100.
        for bad in (-1., float('inf')):
            with self.subTest(root=bad), self.assertRaisesRegex(ValueError, 'finite, nonnegative'):
                evaluate(net, (*batch[:4], torch.tensor([bad, 60., float('nan'), 0.])), 'cpu', value_mix=0.5)
        with self.assertRaisesRegex(ValueError, 'old HP units'):
            evaluate(net, batch, 'cpu', teacher_root_mix=0.5)


class GradientClip(unittest.TestCase):
    def test_opt_in_all_parameter_norm_and_prediction_log(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(1000.))
                self.policy = torch.nn.Parameter(torch.tensor(0.))
            def forward(self, *inputs):
                n = len(inputs[0])
                return self.value.expand(n), torch.stack((self.policy, -self.policy)).expand(n, 2)
        batch = ({name: torch.zeros(2, 1) for name in NAMES}, torch.zeros(2),
                 torch.tensor([[0., 1.], [0., 1.]]), torch.ones(2, dtype=torch.bool),
                 torch.full((2,), float('nan')))
        data = SimpleNamespace(batches=lambda size, rng: iter([batch]))
        def train(clip):
            net = Net(); optimizer = torch.optim.SGD(net.parameters(), lr=1)
            before = torch.stack([p.detach().clone() for p in net.parameters()])
            args = SimpleNamespace(batch=2, device='cpu', policy_temp=1., value_mix=1., teacher_root_mix=1., grad_clip=clip)
            stats = run_epoch(net, data, args, optimizer)
            after = torch.stack([p.detach() for p in net.parameters()])
            return stats, torch.linalg.vector_norm(after - before).item(), net
        off, off_norm, off_net = train(None)
        manual = Net(); optimizer = torch.optim.SGD(manual.parameters(), lr=1)
        value, policy, _, n, n_policy = evaluate(manual, batch, 'cpu')
        optimizer.zero_grad(); (value / n + policy / n_policy).backward(); optimizer.step()
        for name, parameter in manual.state_dict().items():
            self.assertTrue(torch.equal(parameter, off_net.state_dict()[name]))
        clipped, clipped_norm, net = train(.05)
        self.assertGreater(off_norm, 1)
        self.assertLessEqual(clipped_norm, .0501)
        self.assertEqual(off['max_abs_value_prediction'], 1000)
        self.assertEqual(clipped['max_abs_value_prediction'], 1000)
        args = SimpleNamespace(batch=2, device='cpu', policy_temp=1., value_mix=1., teacher_root_mix=1., grad_clip=.05)
        val = run_epoch(net, data, args)
        self.assertEqual(val['max_abs_value_prediction'], net.value.detach().abs().item())
    def test_bad_clip_rejected_before_data_read(self):
        for clip in ('0', '-1', 'nan', 'inf'):
            with self.subTest(clip=clip), patch('sys.argv', ['train', '--data', '/missing', '--out', '/missing', '--grad-clip', clip]):
                with self.assertRaises(SystemExit): main()


if __name__ == '__main__':
    unittest.main()
