"""Typed access preserves the native numeric token contract."""
import unittest

import torch

from .tokens import CardZone, TokenKind, TokenRows, WIDTH


class TokenTests(unittest.TestCase):
    def test_native_fields_and_kind_specific_zones(self):
        # Raw wire values deliberately pin the shared C++/Python encoding contract.
        data = torch.zeros(8, WIDTH)
        data[:, :4] = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 1], [2, 17, 0, 0],
                                   [2, 19, 0, 1], [2, 23, 0, 2], [2, 29, 0, 3], [2, 37, 0, 4], [5, 31, 0, 1]])
        before = data.clone()
        rows = TokenRows(data)
        self.assertEqual(rows.valid.tolist(), [False, True, True, True, True, True, True, True])
        self.assertEqual(rows.is_kind(TokenKind.RELIC).nonzero().flatten().tolist(), [7])
        for zone, index in [(CardZone.HAND, 2), (CardZone.DRAW, 3),
                            (CardZone.DISCARD, 4), (CardZone.EXHAUST, 5), (CardZone.OFFERED, 6)]:
            self.assertEqual(rows.in_card_zone(zone).nonzero().flatten().tolist(), [index])
        self.assertEqual(rows.id.tolist(), [0, 0, 17, 19, 23, 29, 37, 31])
        torch.testing.assert_close(rows.numeric, data[:, 3:])
        torch.testing.assert_close(data, before, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
