"""The same observable prediction contract applies to both architectures."""
import unittest

import torch

from ._test_support import combat_tokens, small_spec
from .model import ModelSpec, build_model
from .tokens import TokenKind, TokenRows, WIDTH


class ModelContract:
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.model = build_model(small_spec(self.kind)).eval()
        self.tokens = combat_tokens()

    def test_values_are_finite_probabilities(self):
        values, _ = self.model(self.tokens)

        self.assertEqual(values.shape, (len(self.tokens),))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(((0 <= values) & (values <= 1)).all())

    def test_only_legal_action_positions_receive_policy_scores(self):
        _, logits = self.model(self.tokens)
        actions = TokenRows(self.tokens).is_kind(TokenKind.ACTION)

        self.assertEqual(logits.shape, self.tokens.shape[:2])
        self.assertTrue(torch.isfinite(logits[actions]).all())
        self.assertTrue((logits[~actions] < -1e8).all())
        policy = logits.softmax(dim=1)
        self.assertTrue((policy[~actions] == 0).all())

    def test_padding_does_not_change_real_token_predictions(self):
        padding = torch.zeros(len(self.tokens), 5, WIDTH)
        padded = torch.cat([self.tokens, padding], dim=1)
        expected_values, expected_logits = self.model(self.tokens)

        actual_values, actual_logits = self.model(padded)

        torch.testing.assert_close(actual_values, expected_values)
        torch.testing.assert_close(actual_logits[:, :self.tokens.shape[1]], expected_logits)

    def test_autocast_keeps_predictions_finite_and_float32(self):
        with torch.autocast('cpu', torch.bfloat16):
            values, logits = self.model(self.tokens)

        self.assertEqual(values.dtype, torch.float32)
        self.assertEqual(logits.dtype, torch.float32)
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.isfinite(logits).all())


class TransformerTests(ModelContract, unittest.TestCase):
    kind = 'transformer'


class PooledTransformerTests(ModelContract, unittest.TestCase):
    kind = 'entity'


class ArchitectureValidationTests(unittest.TestCase):
    def test_unknown_architecture_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelSpec('unknown', dim=16, layers=1, heads=2)

    def test_attention_width_must_be_divisible_by_head_count(self):
        with self.assertRaises(ValueError):
            ModelSpec('transformer', dim=15, layers=1, heads=2)

    def test_architecture_dimensions_must_be_positive(self):
        for invalid in [dict(dim=0), dict(layers=0), dict(heads=0)]:
            dimensions = dict(dim=16, layers=1, heads=2) | invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ModelSpec('transformer', **dimensions)


if __name__ == '__main__':
    unittest.main()
