"""Entity model behavior, and the model contract every kind must keep."""
import unittest

import torch

from agents.combat.nn.entity.encoding import CardZone
from agents.combat.nn.model import AUXILIARY, LOSS_BINS, WIN_BINS, ModelSpec, build_model
from agents.combat.nn.tokens import TokenKind, TokenRows, WIDTH


def small_spec(value='win'):
    return ModelSpec('entity', dim=16, layers=1, heads=2, value=value)


def combat_tokens():
    kinds = [TokenKind.CONTEXT, TokenKind.MONSTER, TokenKind.RELIC,
             TokenKind.CARD, TokenKind.CARD, TokenKind.CARD, TokenKind.CARD,
             TokenKind.ACTION, TokenKind.ACTION, TokenKind.ACTION]
    tokens = torch.zeros(2, len(kinds), WIDTH)
    tokens[..., 0] = torch.tensor(kinds)
    tokens[..., 1] = torch.arange(len(kinds))
    tokens[:, 4:7, 3] = torch.tensor([CardZone.DRAW, CardZone.DISCARD, CardZone.EXHAUST])
    tokens[1, 3, 3] = CardZone.OFFERED
    tokens[..., 4:] = torch.rand(2, len(kinds), WIDTH - 4)
    return tokens



def target_tokens():
    """Two same-species monsters and actions that differ only in which one they target, plus that scene permuted."""
    tokens = torch.zeros(1, 8, WIDTH)
    tokens[0, :, 0] = torch.tensor([1, 3, 3, 6, 6, 6, 6, 6])
    tokens[0, 1:3, 1] = 17
    tokens[0, 1:3, 3] = torch.tensor([.2, .8])
    tokens[0, 3:5, 1:3] = torch.tensor([23, 1])  # non-damaging card
    tokens[0, 5:7, 1:3] = torch.tensor([11, 2])  # targeted potion
    tokens[0, 3:7, -1] = torch.tensor([1, 2, 1, 2])
    tokens[0, 7, 2] = 5  # no-target End Turn
    permuted = tokens[:, [0, 2, 1, 3, 4, 5, 6, 7]].clone()
    permuted[0, 3:7, -1] = torch.tensor([2, 1, 2, 1])
    return tokens, permuted

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_policy_is_a_probability_over_legal_moves_only(self):
        for value in ('win', 'outcome'):
            with self.subTest(value=value):
                tokens = combat_tokens()
                values, logits = build_model(small_spec(value)).eval()(tokens)
                actions = TokenRows(tokens).is_kind(TokenKind.ACTION)

                self.assertEqual(values.shape, (len(tokens),))
                self.assertEqual(logits.shape, tokens.shape[:2])
                self.assertTrue(torch.isfinite(values).all())
                self.assertTrue(((0 <= values) & (values <= 1)).all())
                self.assertTrue(torch.isfinite(logits[actions]).all())
                self.assertTrue((logits.softmax(dim=1)[~actions] == 0).all())

    def test_padding_does_not_change_predictions(self):
        for value in ('win', 'outcome'):
            with self.subTest(value=value):
                model = build_model(small_spec(value)).eval()
                tokens = combat_tokens()
                padding = torch.zeros(len(tokens), 5, WIDTH)
                expected_values, expected_logits = model(tokens)

                values, logits = model(torch.cat([tokens, padding], dim=1))

                torch.testing.assert_close(values, expected_values)
                torch.testing.assert_close(logits[:, :tokens.shape[1]], expected_logits)

    def test_training_speedup_predicts_the_same_as_the_exported_path(self):
        for value in ('win', 'outcome'):
            with self.subTest(value=value):
                model = build_model(small_spec(value)).eval()
                tokens = combat_tokens()
                tokens = torch.cat([tokens, torch.zeros(len(tokens), 5, WIDTH)], dim=1)

                for actual, expected in zip(model(tokens), model.predict(tokens)):
                    torch.testing.assert_close(actual, expected)

    def test_win_probability_is_the_mass_of_the_win_bins(self):
        for auxiliary in (False, True):
            with self.subTest(auxiliary=auxiliary):
                spec = ModelSpec('entity', dim=16, layers=1, heads=2, value='outcome', auxiliary=auxiliary)
                model = build_model(spec).eval()
                tokens = combat_tokens()
                context, _ = model.heads(tokens)
                values, _ = model(tokens)
                bins = LOSS_BINS + WIN_BINS
                self.assertEqual(context.shape, (len(tokens), bins + len(AUXILIARY) * auxiliary))
                torch.testing.assert_close(values, context[:, :bins].softmax(-1)[:, LOSS_BINS:].sum(-1))
    def test_moves_stay_linked_to_their_target_monster(self):
        model = build_model(small_spec()).eval()
        tokens, permuted = target_tokens()
        values, logits = model(tokens)
        # Swapping the two identical monsters (and the targets that point at them) changes nothing...
        for actual, expected in zip(model(permuted), (values, logits)):
            torch.testing.assert_close(actual, expected)
        # ...yet the same move aimed at different monsters is scored differently.
        self.assertGreater(abs((logits[0, 3] - logits[0, 4]).item()), 1e-5)
        self.assertGreater(abs((logits[0, 5] - logits[0, 6]).item()), 1e-5)

    def test_attention_overflow_is_nan_so_the_worker_falls_back_to_the_full_model(self):
        model = build_model(small_spec()).eval()
        tokens = combat_tokens()
        values, _ = model.deployed(attended=1)(tokens)
        self.assertTrue(torch.isnan(values).all())
        self.assertTrue(torch.isfinite(model(tokens)[0]).all())


if __name__ == '__main__':
    unittest.main()
