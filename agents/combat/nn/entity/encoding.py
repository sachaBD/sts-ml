"""The entity model's token layout (the "legacy" encoding, entity/encoding.cpp)."""
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from agents.combat.nn.tokens import TokenKind, TokenRows

KIND_CAPACITY = 8


class CardZone(IntEnum):
    HAND = 0
    DRAW = 1
    DISCARD = 2
    EXHAUST = 3
    OFFERED = 4


@dataclass(frozen=True)
class EntityRows(TokenRows):
    @property
    def numeric(self) -> Tensor:
        # The final action column is a row reference, not a numeric feature.
        return torch.cat((self.data[..., 3:-1], self.data[..., -1:].masked_fill(
            self.is_kind(TokenKind.ACTION)[..., None], 0)), dim=-1)

    def with_targets(self, embedded: Tensor) -> Tensor:
        """Add the target monster embedding before attention; row zero means no target."""
        index = (self.data[..., -1] * self.is_kind(TokenKind.ACTION)).long()
        target = embedded.gather(-2, index[..., None].expand_as(embedded))
        return embedded + target * (index > 0)[..., None]

    def in_card_zone(self, zone: CardZone) -> Tensor:
        return self.is_kind(TokenKind.CARD) & ((self.numeric[..., 0] - zone).abs() < 0.5)
