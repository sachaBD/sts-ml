"""Separate supervision must preserve each objective's mean and one shared optimizer update."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .model import LOSS_BINS, WIN_BINS, ModelSpec
from .replay import Replay, TARGETS
from .test_replay import samples
from .tokens import WIDTH
from .train import losses, outcome_bin, train


class Tiny(torch.nn.Module):
    spec = ModelSpec('transformer', 1, 1, 1)

    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Parameter(torch.tensor(.2))
        self.value = torch.nn.Parameter(torch.tensor(.3))
        self.policy = torch.nn.Parameter(torch.tensor(.4))

    def heads(self, tokens):
        hidden = self.trunk * tokens[:, :, 3]
        return self.value * hidden[:, :1], (self.policy * hidden).masked_fill(tokens[:, :, 0] != 6, -1e9)


def batch(size, scale):
    tokens = torch.zeros(size, 3, WIDTH)
    tokens[:, :, 0] = torch.tensor([1, 6, 6])
    tokens[:, :, 3] = scale * torch.arange(1, size + 1)[:, None] * torch.tensor([1., 2., -1.])
    policy = torch.tensor([0., .25, .75]).expand(size, -1)
    targets = torch.zeros(size, len(TARGETS))
    targets[:, TARGETS.index('won')] = torch.arange(size).remainder(2).float()
    return tokens, policy, targets


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
                    with patch.object(replay, 'batches', side_effect=batches):
                        logged = train(actual, optimizer, replay, None, None, 1, 5, 'cpu',
                                       log=lambda _: None, value_batch=3)
                    torch.testing.assert_close(logged, (v.item(), p.item(), v.item()))
                    for name, parameter in actual.named_parameters():
                        torch.testing.assert_close(parameter, dict(expected.named_parameters())[name])


class OutcomeBinTests(unittest.TestCase):
    def test_bins_order_every_loss_below_every_win_by_margin(self):
        torch.manual_seed(0)
        targets = torch.rand(4096, len(TARGETS))
        targets[:, TARGETS.index('won')] = (targets[:, 0] < .5).float()
        targets[:2, TARGETS.index('hp')], targets[2:4, TARGETS.index('enemy_hp')] = 1., torch.tensor([0., 1.])
        bins = outcome_bin(targets)
        won = targets[:, TARGETS.index('won')] > 0
        self.assertTrue(((0 <= bins) & (bins < LOSS_BINS + WIN_BINS)).all())
        self.assertTrue(torch.equal(bins >= LOSS_BINS, won))
        for margin, mask in ((targets[:, TARGETS.index('hp')], won), (-targets[:, TARGETS.index('enemy_hp')], ~won)):
            order = margin[mask].argsort()
            self.assertTrue((bins[mask][order].diff() >= 0).all())


if __name__ == '__main__':
    unittest.main()
