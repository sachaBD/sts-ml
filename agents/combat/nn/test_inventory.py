"""Inventory predictions preserve meaning under relabeling, padding and export."""
from pathlib import Path
import tempfile
import unittest

import torch

from .export import ExportSpec, ModelExporter
from .inventory_transformer import (ACTION, CARD_ACTION, CARD, CONTEXT, MONSTER, MONSTER_EFFECT,
                                    OWNER, SOURCE, TARGET, VERSION)
from .model import ModelSpec, build_model
from .test_export import model_predictions, onnx_predictions
from .tokens import WIDTH


def inventory_tokens():
    kinds = [CONTEXT, CARD, MONSTER, MONSTER, MONSTER_EFFECT, ACTION, ACTION]
    tokens = torch.zeros(2, len(kinds), WIDTH)
    tokens[..., 0] = torch.tensor(kinds)
    tokens[..., 1] = torch.arange(len(kinds))
    tokens[..., 3:20] = torch.rand(2, len(kinds), 17)
    tokens[:, 0, 2] = VERSION
    tokens[:, 4, OWNER] = 3  # first monster
    tokens[:, 5:, SOURCE] = 2  # card
    tokens[:, 5:, TARGET] = torch.tensor([3, 4])
    tokens[:, 5:, 1:3] = torch.tensor([1, CARD_ACTION])
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
        self.model = build_model(ModelSpec('inventory_v1', dim=16, layers=1, heads=2)).eval()
        self.tokens = inventory_tokens()

    def test_padding_does_not_change_real_predictions(self):
        expected_values, expected_logits = self.model(self.tokens)
        padded = torch.cat([self.tokens, torch.zeros(2, 5, WIDTH)], dim=1)
        values, logits = self.model(padded)
        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(logits[:, :self.tokens.shape[1]], expected_logits)

    def test_consistent_entity_relabeling_only_reorders_action_scores(self):
        values, logits = self.model(self.tokens)
        order = torch.tensor([0, 4, 3, 1, 2, 6, 5])
        actual_values, actual_logits = self.model(permuted(self.tokens, order))
        torch.testing.assert_close(actual_values, values)
        torch.testing.assert_close(actual_logits, logits[:, order])

    def test_wrong_replay_schema_cannot_produce_a_valid_value(self):
        legacy = self.tokens.clone()
        legacy[:, 0, 2] = 0
        values, _ = self.model(legacy)
        self.assertTrue(torch.isnan(values).all())

    def test_export_preserves_predictions(self):
        padded = torch.cat([self.tokens, torch.zeros(2, 3, WIDTH)], dim=1)
        exporter = ModelExporter(self.model.spec, ExportSpec(tokens=padded.shape[1]))
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            exporter.export(self.model, directory, batch=2, bank=())
            for name, inputs in [('model.onnx', self.tokens[:1]), ('model-gpu.onnx', padded.half())]:
                with self.subTest(artifact=name):
                    torch.testing.assert_close(onnx_predictions(directory / name, inputs),
                                               model_predictions(self.model, inputs.float()), rtol=3e-4, atol=3e-5)


if __name__ == '__main__':
    unittest.main()
