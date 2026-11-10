"""Separate supervision must preserve each objective's mean and one shared optimizer update."""
import copy
from itertools import count
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

from agents.combat.nn.model import LOSS_BINS, WIN_BINS, ModelSpec, build_model
from agents.combat.nn.training.replay import Replay, TARGETS
from agents.combat.nn.training.test_replay import samples
from agents.combat.nn.tokens import WIDTH
from agents.combat.nn.training.train import AUXILIARY_WEIGHT, LOSSES, losses, outcome_bin, train


class Tiny(torch.nn.Module):
    spec = ModelSpec('entity', 1, 1, 1)

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
            with patch('agents.combat.nn.training.train.time', SimpleNamespace(monotonic=lambda: next(clock))):
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
            runs[1][0][:2], atol=1e-4, rtol=1e-4)

    def test_independent_losses_match_explicit_objective_for_any_microbatch_partition(self):
        policy_batch, value_batch = batch(5, 2.), batch(3, -3.)
        torch.manual_seed(0)
        nets = dict(tiny=Tiny, auxiliary=lambda: build_model(ModelSpec('entity', 8, 1, 1, 'outcome', True)))
        # Tiny checks every partition; the real model's bf16 matmuls round differently per microbatch shape.
        cases = [('tiny', policy_sizes, value_sizes)
                 for policy_sizes, value_sizes in [([5], [3]), ([2, 3], [1, 2]), ([1, 1, 3], [2, 1])]]
        cases.append(('auxiliary', [5], [3]))
        for net, policy_sizes, value_sizes in cases:
            with (self.subTest(net=net, policy=policy_sizes, value=value_sizes), tempfile.TemporaryDirectory() as tmp):
                path = samples(Path(tmp) / 'a', [('a', 1, 1, 1), ('a', 1, 2, 0), ('b', 1, 3, 1)])
                with Replay([path], balanced=False) as replay:
                    expected = nets[net]()
                    actual = copy.deepcopy(expected)
                    optimizer = torch.optim.SGD(actual.parameters(), lr=.1)
                    reference_optimizer = torch.optim.SGD(expected.parameters(), lr=.1)
                    value_side = dict(zip(LOSSES, losses(expected, *value_batch, 'cpu')))
                    policy_side = dict(zip(LOSSES, losses(expected, *policy_batch, 'cpu')))
                    v, p, w = value_side['value'], policy_side['policy'], value_side['win']
                    turns, round_ = value_side['turns'], policy_side['round']
                    (v + p + AUXILIARY_WEIGHT * (turns + round_)).backward()
                    torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.)
                    reference_optimizer.step()
                    def batches(size, *, combat_balanced=False):
                        return iter(chunks(value_batch, value_sizes) if combat_balanced else chunks(policy_batch, policy_sizes))
                    with patch.object(replay, 'batches', side_effect=batches):
                        logged = train(actual, optimizer, replay, None, None, 1, 5, 'cpu',
                                       log=lambda _: None, value_batch=3)
                    torch.testing.assert_close(logged, tuple(x.item() for x in (v, p, w, round_, turns)))
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
