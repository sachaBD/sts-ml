"""Separate supervision must preserve each objective's mean and one shared optimizer update."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .replay import Replay
from .test_replay import samples
from .tokens import WIDTH
from .train import losses, train


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Parameter(torch.tensor(.2))
        self.value = torch.nn.Parameter(torch.tensor(.3))
        self.policy = torch.nn.Parameter(torch.tensor(.4))

    def forward(self, tokens):
        hidden = self.trunk * tokens[:, :, 3]
        return (self.value * hidden[:, 0]).sigmoid(), (self.policy * hidden).masked_fill(tokens[:, :, 0] != 6, -1e9)


def batch(size, scale):
    tokens = torch.zeros(size, 3, WIDTH)
    tokens[:, :, 0] = torch.tensor([1, 6, 6])
    tokens[:, :, 3] = scale * torch.arange(1, size + 1)[:, None] * torch.tensor([1., 2., -1.])
    policy = torch.tensor([0., .25, .75]).expand(size, -1)
    won = torch.arange(size).remainder(2).float()
    return tokens, policy, won


def chunks(tensors, sizes):
    return list(zip(*(tensor.split(sizes) for tensor in tensors)))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_independent_losses_match_explicit_objective_for_any_microbatch_partition(self):
        policy_batch, value_batch = batch(5, 2.), batch(3, -3.)
        for policy_sizes, value_sizes in [([5], [3]), ([2, 3], [1, 2]), ([1, 1, 3], [2, 1])]:
            with self.subTest(policy=policy_sizes, value=value_sizes), tempfile.TemporaryDirectory() as tmp:
                path = samples(Path(tmp) / 'a', [('a', 1, 1, 1), ('a', 1, 2, 0), ('b', 1, 3, 1)])
                with Replay([path], balanced=False) as replay:
                    expected, actual = Tiny(), Tiny()
                    optimizer = torch.optim.SGD(actual.parameters(), lr=.1)
                    reference_optimizer = torch.optim.SGD(expected.parameters(), lr=.1)
                    v = losses(expected, *value_batch, 'cpu')[0]
                    p = losses(expected, *policy_batch, 'cpu')[1]
                    (v + p).backward()
                    torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.)
                    reference_optimizer.step()
                    def batches(size, *, combat_balanced=False):
                        return iter(chunks(value_batch, value_sizes) if combat_balanced else chunks(policy_batch, policy_sizes))
                    with patch.object(replay, 'batches', side_effect=batches), \
                         patch.object(optimizer, 'step', wraps=optimizer.step) as update, \
                         patch('torch.nn.utils.clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
                        logged = train(actual, optimizer, replay, None, None, 1, 5, 'cpu',
                                       log=lambda _: None, value_batch=3)
                    self.assertEqual(update.call_count, 1)
                    self.assertEqual(clip.call_count, 1)
                    torch.testing.assert_close(logged, (v.item(), p.item()))
                    for name, parameter in actual.named_parameters():
                        torch.testing.assert_close(parameter, dict(expected.named_parameters())[name])

    def test_invalid_value_batch_fails_before_any_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = samples(Path(tmp) / 'a', [('a', 2, 1, 1)])
            with Replay([path], balanced=False) as replay:
                for value_batch in (-1, 2):
                    net = Tiny()
                    before = {name: p.detach().clone() for name, p in net.named_parameters()}
                    optimizer = torch.optim.AdamW(net.parameters())
                    with self.assertRaises(ValueError):
                        train(net, optimizer, replay, None, None, 1, 2, 'cpu', value_batch=value_batch)
                    self.assertEqual(optimizer.state, {})
                    for name, p in net.named_parameters():
                        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
            net = Tiny()
            with self.assertRaisesRegex(ValueError, 'requires Replay'):
                train(net, torch.optim.SGD(net.parameters(), lr=.1), *batch(2, 1.), 1, 2, 'cpu', value_batch=1)


if __name__ == '__main__':
    unittest.main()
