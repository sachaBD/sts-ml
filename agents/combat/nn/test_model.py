"""Behavior shared by both supported combat models."""
import unittest

import torch

from ._test_support import combat_tokens, small_spec
from .model import ModelSpec, build_model
from .tokens import TokenKind, TokenRows, WIDTH


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_predictions_are_probabilities_over_legal_actions(self):
        for kind in ('transformer', 'entity'):
            with self.subTest(kind=kind):
                tokens = combat_tokens()
                values, logits = build_model(small_spec(kind)).eval()(tokens)
                actions = TokenRows(tokens).is_kind(TokenKind.ACTION)

                self.assertEqual(values.shape, (len(tokens),))
                self.assertEqual(logits.shape, tokens.shape[:2])
                self.assertTrue(torch.isfinite(values).all())
                self.assertTrue(((0 <= values) & (values <= 1)).all())
                self.assertTrue(torch.isfinite(logits[actions]).all())
                self.assertTrue((logits.softmax(dim=1)[~actions] == 0).all())

    def test_padding_does_not_change_predictions(self):
        for kind in ('transformer', 'entity'):
            with self.subTest(kind=kind):
                model = build_model(small_spec(kind)).eval()
                tokens = combat_tokens()
                padding = torch.zeros(len(tokens), 5, WIDTH)
                expected_values, expected_logits = model(tokens)

                values, logits = model(torch.cat([tokens, padding], dim=1))

                torch.testing.assert_close(values, expected_values)
                torch.testing.assert_close(logits[:, :tokens.shape[1]], expected_logits)

    def test_entity_forward_matches_attention_over_every_position(self):
        model = build_model(small_spec('entity')).eval()
        tokens = combat_tokens()
        tokens = torch.cat([tokens, torch.zeros(len(tokens), 5, WIDTH)], dim=1)

        for actual, expected in zip(model(tokens), model.predict(tokens)):
            torch.testing.assert_close(actual, expected)

    def test_invalid_architectures_are_rejected(self):
        for invalid in [dict(kind='unknown'), dict(dim=0), dict(layers=0), dict(heads=0), dict(dim=15)]:
            spec = dict(kind='transformer', dim=16, layers=1, heads=2) | invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ModelSpec(**spec)


if __name__ == '__main__':
    unittest.main()
