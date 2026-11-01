"""Typed access must interpret the native wire format without changing its meaning."""
import unittest

import torch

from .tokens import CardZone, TokenKind, TokenRows, WIDTH


class TokenTests(unittest.TestCase):
    def test_fields_decode_the_native_wire_layout(self):
        wire = torch.zeros(1, WIDTH)
        wire[0, :4] = torch.tensor([2, 17, 23, 1])  # card, primary ID, secondary ID, draw zone

        rows = TokenRows(wire)

        self.assertTrue(rows.is_kind(TokenKind.CARD).item())
        self.assertEqual(rows.id.item(), 17)
        self.assertEqual(rows.id2.item(), 23)
        self.assertEqual(rows.numeric.shape[-1], 65)
        self.assertEqual(rows.numeric[0, 0].item(), 1)

    def test_padding_is_not_a_real_token(self):
        wire = torch.zeros(2, WIDTH)
        wire[1, 0] = TokenKind.CONTEXT

        self.assertEqual(TokenRows(wire).valid.tolist(), [False, True])

    def test_card_zones_match_the_native_wire_identifiers(self):
        wire = torch.zeros(5, WIDTH)
        wire[:, 0] = 2  # native card kind
        wire[:, 3] = torch.tensor([0, 1, 2, 3, 4])
        zones = [CardZone.HAND, CardZone.DRAW, CardZone.DISCARD, CardZone.EXHAUST, CardZone.OFFERED]

        for index, zone in enumerate(zones):
            with self.subTest(zone=zone):
                expected = [position == index for position in range(len(zones))]
                self.assertEqual(TokenRows(wire).in_card_zone(zone).tolist(), expected)

    def test_a_relic_numeric_field_is_not_a_card_zone(self):
        wire = torch.zeros(2, WIDTH)
        wire[:, 0] = torch.tensor([TokenKind.CARD, TokenKind.RELIC])
        wire[:, 3] = CardZone.DRAW

        self.assertEqual(TokenRows(wire).in_card_zone(CardZone.DRAW).tolist(), [True, False])


if __name__ == '__main__':
    unittest.main()
