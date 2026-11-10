"""The entity encoding's tokens, as layout.toml's [entity] describes them (entity/encoding.cpp writes them)."""
from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from agents.combat.nn.layout import LAYOUT
from agents.combat.nn.tokens import ROW, TokenRows

ENTITY = LAYOUT.entity
TokenKind = IntEnum('TokenKind', {name.upper(): id_ for name, id_ in vars(ENTITY.kind).items()})
CardZone = IntEnum('CardZone', {name.upper(): value for name, value in vars(ENTITY.card_zone).items()})
TARGET = ENTITY.link.target  # the row's last column (entity/encoding.cpp asserts it)
KIND_CAPACITY = 8  # kind embeddings, part of the architecture


@dataclass(frozen=True)
class EntityRows(TokenRows):
    @property
    def numeric(self) -> Tensor:
        # An action's target column is a row reference, not a numeric feature.
        return torch.cat((self.data[..., ROW.features:TARGET], self.data[..., TARGET:].masked_fill(
            self.is_kind(TokenKind.ACTION)[..., None], 0)), dim=-1)

    def with_targets(self, embedded: Tensor) -> Tensor:
        """Add the target monster embedding before attention; row zero means no target."""
        reference = (self.data[..., TARGET] * self.is_kind(TokenKind.ACTION)).long()
        index = (reference - ENTITY.link_offset).clamp(min=0)
        target = embedded.gather(-2, index[..., None].expand_as(embedded))
        return embedded + target * (reference > 0)[..., None]

    def in_card_zone(self, zone: CardZone) -> Tensor:
        return self.is_kind(TokenKind.CARD) & ((self.numeric[..., 0] - zone).abs() < 0.5)
