"""Separate supervision must preserve each objective's mean and one shared optimizer update."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from agents.combat.nn.model import LOSS_BINS, WIN_BINS, THINKING_ROUNDS, ModelSpec, build_model
from agents.combat.nn.training.replay import Replay, TARGETS
from agents.combat.nn.training.test_replay import samples
from agents.combat.nn.training.train import AUXILIARY_WEIGHT, LOSSES, losses, outcome_bin, train


class Tiny(torch.nn.Module):
    spec = ModelSpec('entity', 1, 1, 1, 'win', False)

    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Parameter(torch.tensor(.2))
        self.value = torch.nn.Parameter(torch.tensor(.3))
        self.policy = torch.nn.Parameter(torch.tensor(.4))

    def readouts(self, tokens):
        hidden = self.trunk * tokens.sum(-1)
        not_action = torch.arange(tokens.shape[1]) == 0  # example_tokens: a context row, then legal actions
        return self.value * hidden[:, :1], (self.policy * hidden).masked_fill(not_action, -1e9), hidden[:, :1].expand(-1, THINKING_ROUNDS)


def batch(size):
    tokens = build_model(Tiny.spec).example_tokens(size, 3)
    policy = torch.tensor([0., .25, .75]).expand(size, -1)
    targets = torch.rand(size, len(TARGETS), generator=torch.Generator().manual_seed(size))
    targets[:, TARGETS.index('won')] = torch.arange(size).remainder(2).float()
    return tokens, policy, targets


def chunks(tensors, sizes):
    return list(zip(*(tensor.split(sizes) for tensor in tensors)))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_losses_do_not_depend_on_how_the_batch_is_split(self):
        policy_batch, value_batch = batch(5), batch(3)  # samples differ in their targets
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
                    torch.testing.assert_close(logged, tuple(x.item() for x in (v, p, w, round_, turns, policy_side['thinking'])))
                    for name, parameter in actual.named_parameters():
                        torch.testing.assert_close(parameter, dict(expected.named_parameters())[name])


    def test_thinking_masks_missing_rounds_and_unprobed_rows(self):
        x, policy, y = batch(3)
        model = Tiny()
        with torch.no_grad():
            model.trunk.zero_()
        gains = torch.full((len(y), THINKING_ROUNDS), torch.nan)
        gains[0, 0], gains[0, 2], gains[2, 0] = -.5, 0., .25
        policy = policy.clone()
        policy[0] = 0  # a cheap probe-only decision still trains the thinking head
        result = losses(model, x, policy, torch.cat([y, gains], dim=1), 'cpu')
        torch.testing.assert_close(result[-1], torch.tensor((.5**2 + .25**2) / 3))
        torch.testing.assert_close(result[:-1], losses(model, x[1:], policy[1:], y[1:], 'cpu')[:-1])
        result[-1].backward()
        self.assertTrue(torch.isfinite(model.trunk.grad))

    def test_all_inference_outputs_use_one_trunk_pass(self):
        for kind in ('entity', 'inventory_v1'):
            net = build_model(ModelSpec(kind, 8, 1, 1, 'win', False)).eval()
            x = net.example_tokens()
            expected = net(x)
            with patch.object(net.encoder, 'forward', wraps=net.encoder.forward) as trunk:
                value, policy, gains = net.predict(x)
                self.assertEqual(trunk.call_count, 1)
            self.assertEqual(gains.shape, (len(x), THINKING_ROUNDS))
            torch.testing.assert_close((value, policy), expected)

    def test_thinking_head_handles_small_k_is_padding_invariant_and_uses_policy_value_summaries(self):
        torch.manual_seed(0)
        for kind in ('entity', 'inventory_v1'):
            net = build_model(ModelSpec(kind, 8, 1, 1, 'win', False)).eval()
            with torch.no_grad():
                net.thinking.net[-1].weight.fill_(0.25)
            for length in (2, 3, 5):  # K = 1, 2, 4 legal actions
                x = net.example_tokens(2, length)
                padded = torch.cat([x, torch.zeros(2, 3, x.shape[-1])], dim=1)
                _, _, g = net.predict(x)
                _, _, g_pad = net.predict(padded)
                self.assertTrue(torch.isfinite(g).all())
                torch.testing.assert_close(g, g_pad, rtol=1e-5, atol=1e-5)
            # Same hidden state with different policy logits produces different thinking output.
            hidden = torch.zeros(2, 8)
            legal = torch.tensor([[False, True, True, True], [False, True, True, True]])
            sharp = torch.tensor([[-1e9, 5., 0., 0.], [-1e9, 5., 0., 0.]])
            flat_ = torch.tensor([[-1e9, 0., 0., 0.], [-1e9, 0., 0., 0.]])
            v = torch.tensor([0.5, 0.5])
            self.assertFalse(torch.allclose(net.thinking(hidden, v, sharp, legal), net.thinking(hidden, v, flat_, legal)))


class OutcomeBinTests(unittest.TestCase):
    def test_outcome_bins_rank_every_loss_below_every_win(self):
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
