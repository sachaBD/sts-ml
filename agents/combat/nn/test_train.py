"""Separate supervision must preserve each objective's mean and one shared optimizer update."""
from itertools import count
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from .replay import Replay, TARGETS
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
    targets = torch.rand(size, len(TARGETS), generator=torch.Generator().manual_seed(size))
    targets[:, TARGETS.index('won')] = torch.arange(size).remainder(2).float()
    return tokens, policy, targets


def chunks(tensors, sizes):
    return list(zip(*(tensor.split(sizes) for tensor in tensors)))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_telemetry_preserves_learning_rng_and_log_windows(self):
        reports, runs = [], []
        for callback in (None, reports.append):
            torch.manual_seed(17)
            net = Tiny()
            optimizer = torch.optim.SGD(net.parameters(), lr=.001)
            clock, logs = count(), []
            with patch('agents.combat.nn.train.time', SimpleNamespace(monotonic=lambda: next(clock))):
                result = train(net, optimizer, *batch(5, 2.), 501, 5, 'cpu',
                               log=logs.append, on_counters=callback)
            runs.append((result, [p.detach().clone() for p in net.parameters()], torch.get_rng_state()))
            self.assertEqual(len(logs), 2)  # the 500-step log and final partial window
        self.assertEqual(runs[0][0], runs[1][0])
        for expected, actual in zip(runs[0][1], runs[1][1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(runs[0][2], runs[1][2], rtol=0, atol=0)
        self.assertEqual(reports[0].step, 0)
        self.assertEqual(reports[-1].step, 501)
        self.assertTrue(all(a.step < b.step and a.seconds < b.seconds for a, b in zip(reports, reports[1:])))
        before = next(r for r in reports if r.step == 500)
        final = reports[-1]
        torch.testing.assert_close(
            (final.value_loss_sum - before.value_loss_sum, final.policy_loss_sum - before.policy_loss_sum),
            runs[1][0], atol=1e-4, rtol=1e-4)

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
                    with patch.object(replay, 'batches', side_effect=batches):
                        logged = train(actual, optimizer, replay, None, None, 1, 5, 'cpu',
                                       log=lambda _: None, value_batch=3)
                    torch.testing.assert_close(logged, (v.item(), p.item()))
                    for name, parameter in actual.named_parameters():
                        torch.testing.assert_close(parameter, dict(expected.named_parameters())[name])


if __name__ == '__main__':
    unittest.main()
