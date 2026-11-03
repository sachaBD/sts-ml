"""Behavior shared by both supported combat models."""
import unittest

import torch

from ._test_support import combat_tokens, small_spec
from .model import AUXILIARY, LOSS_BINS, WIN_BINS, ModelSpec, build_model
from .tokens import TokenKind, TokenRows, WIDTH


SPECS = [(kind, value) for kind in ('transformer', 'entity') for value in ('win', 'outcome')]



class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_predictions_are_probabilities_over_legal_actions(self):
        for kind, value in SPECS:
            with self.subTest(kind=kind, value=value):
                tokens = combat_tokens()
                values, logits = build_model(small_spec(kind, value)).eval()(tokens)
                actions = TokenRows(tokens).is_kind(TokenKind.ACTION)

                self.assertEqual(values.shape, (len(tokens),))
                self.assertEqual(logits.shape, tokens.shape[:2])
                self.assertTrue(torch.isfinite(values).all())
                self.assertTrue(((0 <= values) & (values <= 1)).all())
                self.assertTrue(torch.isfinite(logits[actions]).all())
                self.assertTrue((logits.softmax(dim=1)[~actions] == 0).all())

    def test_padding_does_not_change_predictions(self):
        for kind, value in SPECS:
            with self.subTest(kind=kind, value=value):
                model = build_model(small_spec(kind, value)).eval()
                tokens = combat_tokens()
                padding = torch.zeros(len(tokens), 5, WIDTH)
                expected_values, expected_logits = model(tokens)

                values, logits = model(torch.cat([tokens, padding], dim=1))

                torch.testing.assert_close(values, expected_values)
                torch.testing.assert_close(logits[:, :tokens.shape[1]], expected_logits)

    def test_entity_forward_matches_attention_over_every_position(self):
        for value in ('win', 'outcome'):
            with self.subTest(value=value):
                model = build_model(small_spec('entity', value)).eval()
                tokens = combat_tokens()
                tokens = torch.cat([tokens, torch.zeros(len(tokens), 5, WIDTH)], dim=1)

                for actual, expected in zip(model(tokens), model.predict(tokens)):
                    torch.testing.assert_close(actual, expected)

    def test_outcome_win_probability_is_the_win_bin_mass(self):
        for kind, auxiliary in [(kind, auxiliary) for kind in ('transformer', 'entity') for auxiliary in (False, True)]:
            with self.subTest(kind=kind, auxiliary=auxiliary):
                spec = ModelSpec(kind, dim=16, layers=1, heads=2, value='outcome', auxiliary=auxiliary)
                model = build_model(spec).eval()
                tokens = combat_tokens()
                context, _ = model.heads(tokens)
                values, _ = model(tokens)
                bins = LOSS_BINS + WIN_BINS
                self.assertEqual(context.shape, (len(tokens), bins + len(AUXILIARY) * auxiliary))
                torch.testing.assert_close(values, context[:, :bins].softmax(-1)[:, LOSS_BINS:].sum(-1))

    def test_invalid_architectures_are_rejected(self):
        for invalid in [dict(kind='unknown'), dict(dim=0), dict(layers=0), dict(heads=0), dict(dim=15), dict(value='hp')]:
            spec = dict(kind='transformer', dim=16, layers=1, heads=2) | invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ModelSpec(**spec)


if __name__ == '__main__':
    unittest.main()
