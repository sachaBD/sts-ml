"""The native wire format determines field and zone meanings."""
import unittest

import torch

from .tokens import CardZone, TokenKind, TokenRows, WIDTH


class TokenTests(unittest.TestCase):
    def test_native_fields_are_decoded_without_reinterpretation(self):
        wire = torch.zeros(1, WIDTH)
        wire[0, :4] = torch.tensor([2, 17, 23, 4])  # card, primary ID, secondary ID, offered zone
        rows = TokenRows(wire)

        self.assertTrue(rows.is_kind(TokenKind.CARD).item())
        self.assertEqual(rows.id.item(), 17)
        self.assertEqual(rows.id2.item(), 23)
        self.assertTrue(rows.in_card_zone(CardZone.OFFERED).item())

    def test_only_cards_can_belong_to_card_zones(self):
        wire = torch.zeros(2, WIDTH)
        wire[:, 0] = torch.tensor([TokenKind.CARD, TokenKind.RELIC])
        wire[:, 3] = CardZone.DRAW

        self.assertEqual(TokenRows(wire).in_card_zone(CardZone.DRAW).tolist(), [True, False])


if __name__ == '__main__':
    unittest.main()
