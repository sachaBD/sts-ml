"""Explicit entities preserve their meaning under padding and consistent relabeling."""
from pathlib import Path
import tempfile
import unittest

import torch

from .export import ExportSpec, ModelExporter
from .inventory_transformer import (ACTION, CARD, CONTEXT, INVENTORY, MONSTER, MONSTER_EFFECT,
                                    OWNER, PLAYER, PLAYER_EFFECT, POTION, RELIC, SOURCE, TARGET)
from .learner import Learner
from .model import AUXILIARY, LOSS_BINS, WIN_BINS, ModelSpec, build_model
from .test_export import model_predictions, onnx_predictions
from .tokens import WIDTH
from .train import losses


def inventory_tokens():
    kinds = [CONTEXT, PLAYER, PLAYER_EFFECT, CARD, INVENTORY, MONSTER, MONSTER,
             MONSTER_EFFECT, POTION, RELIC, ACTION, ACTION, ACTION]
    tokens = torch.zeros(2, len(kinds), WIDTH)
    tokens[..., 0] = torch.tensor(kinds)
    tokens[..., 1] = torch.arange(len(kinds))
    tokens[..., 3:20] = torch.rand(2, len(kinds), 17)
    tokens[:, 0, 2] = 1  # inventory_v1 context marker
    tokens[:, 2, OWNER] = 2  # player
    tokens[:, 7, OWNER] = 6  # first monster
    tokens[:, 10:12, SOURCE] = 4  # hand card
    tokens[:, 10, TARGET], tokens[:, 11, TARGET] = 6, 7
    tokens[:, 12, SOURCE] = 9  # potion
    tokens[:, 12, TARGET] = 7
    tokens[:, 10:12, 1:3] = torch.tensor([3, 1])  # same card identity and action kind
    tokens[:, 10:12, 3:65] = tokens[:, 10:11, 3:65]
    tokens[:, 12, 1:3] = torch.tensor([8, 2])
    return tokens


def permuted(tokens, order):
    inverse = torch.argsort(order)
    result = tokens[:, order].clone()
    for column in (OWNER, SOURCE, TARGET):
        references = result[..., column].long()
        result[..., column] = torch.where(references > 0, inverse[(references - 1).clamp(min=0)] + 1, 0)
    return result


class InventoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.spec = ModelSpec('inventory_v1', dim=16, layers=2, heads=2, value='outcome', auxiliary=True)
        self.model = build_model(self.spec).eval()
        self.tokens = inventory_tokens()

    def test_all_existing_heads_and_training_objectives_are_retained(self):
        context, logits = self.model.heads(self.tokens)
        values, _ = self.model(self.tokens)
        self.assertEqual(context.shape, (2, LOSS_BINS + WIN_BINS + len(AUXILIARY)))
        torch.testing.assert_close(values, context[:, :20].softmax(-1)[:, LOSS_BINS:].sum(-1))
        actions = self.tokens[..., 0] == ACTION
        self.assertTrue((logits.softmax(-1)[~actions] == 0).all())
        policy = actions.float() / actions.sum(-1, keepdim=True)
        targets = torch.tensor([[1., .5, 0., .1, .2, 3.], [0., 0., .4, .3, .0, 0.]])
        measured = losses(self.model, self.tokens, policy, targets, 'cpu')
        self.assertTrue(torch.isfinite(measured).all())
        measured.sum().backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue((self.model.value.weight.grad.abs().sum(-1) > 0).all())

    def test_win_head_remains_supported(self):
        model = build_model(ModelSpec('inventory_v1', dim=16, layers=1, heads=2, value='win', auxiliary=True))
        context, _ = model.heads(self.tokens)
        values, _ = model(self.tokens)
        self.assertEqual(context.shape[-1], 1 + len(AUXILIARY))
        torch.testing.assert_close(values, context[:, 0].sigmoid())

    def test_padding_and_consistent_entity_reordering_preserve_predictions(self):
        expected = self.model(self.tokens)
        padded = torch.cat([self.tokens, torch.zeros(2, 5, WIDTH)], dim=1)
        actual = self.model(padded)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1][:, :self.tokens.shape[1]], expected[1])
        order = torch.tensor([0, 9, 6, 4, 8, 7, 3, 2, 1, 5, 12, 10, 11])
        actual = self.model(permuted(self.tokens, order))
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1][:, order])

    def test_target_and_effect_ownership_affect_predictions(self):
        tokens = self.tokens.clone()
        # Otherwise identical monsters differ only in which one owns the effect.
        tokens[:, 6, :65] = tokens[:, 5, :65]
        _, logits = self.model(tokens)
        self.assertFalse(torch.allclose(logits[:, 10], logits[:, 11]))
        changed = tokens.clone()
        changed[:, 7, OWNER] = 7
        changed[:, 12, TARGET] = 6
        actual = self.model(changed)
        torch.testing.assert_close(actual[1][:, 10], logits[:, 11])
        torch.testing.assert_close(actual[1][:, 11], logits[:, 10])

    def test_legacy_replay_is_not_silently_accepted(self):
        legacy = self.tokens.clone()
        legacy[:, 0, 2] = 0
        context, _ = self.model.heads(legacy)
        self.assertTrue(torch.isnan(context).all())

    def test_checkpoint_roundtrip_keeps_architecture_and_predictions(self):
        learner = Learner(self.spec, lr=1e-3, device='cpu')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.pt'
            learner.save(path)
            restored = Learner(self.spec, lr=1e-3, device='cpu')
            restored.load(path)
            torch.testing.assert_close(restored.net.eval()(self.tokens), learner.net.eval()(self.tokens))
            legacy = Learner(ModelSpec('entity', dim=16, layers=2, heads=2, value='outcome', auxiliary=True),
                             lr=1e-3, device='cpu')
            with self.assertRaises(ValueError):
                legacy.load(path)

    def test_dynamic_and_fixed_exports_preserve_relations_and_current_weights(self):
        capacity = self.tokens.shape[1] + 3
        padded = torch.cat([self.tokens, torch.zeros(2, 3, WIDTH)], dim=1)
        exporter = ModelExporter(self.spec, ExportSpec(tokens=capacity), reuse=True)
        order = torch.tensor([0, 9, 6, 4, 8, 7, 3, 2, 1, 5, 12, 10, 11])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for version in range(2):
                exporter.export(self.model, directory, batch=2, bank=(1,))
                for name, inputs in [('model.onnx', permuted(self.tokens[:1], order)),
                                     ('model-gpu.onnx', padded.half()),
                                     ('model-gpu-b1.onnx', padded[:1].half())]:
                    torch.testing.assert_close(onnx_predictions(directory / name, inputs),
                                               model_predictions(self.model, inputs.float()), rtol=3e-4, atol=3e-5)
                with torch.no_grad():
                    for parameter in self.model.parameters():
                        parameter.add_(.001)
            self.assertFalse(exporter.deployment.attended)


if __name__ == '__main__':
    unittest.main()
