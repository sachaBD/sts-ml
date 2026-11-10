"""The entity encoding's tokens, as entity/layout.toml describes them (entity/encoding.cpp writes them)."""
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import torch
from torch import Tensor

from agents.combat.nn.layout import load
from agents.combat.nn.tokens import FEATURES, TokenRows

LAYOUT = load(Path(__file__).with_name('layout.toml'))
TokenKind = IntEnum('TokenKind', {name.upper(): kind.id for name, kind in LAYOUT.kinds.items()})
CardZone = IntEnum('CardZone', {name.upper(): value for name, value in LAYOUT.card_zones.items()})
WIDTH, ID_CAPACITY, TARGET = LAYOUT.width, LAYOUT.id_capacity, LAYOUT.links['target']
KIND_CAPACITY = 8  # kind embeddings, part of the architecture
assert max(TokenKind) < KIND_CAPACITY
assert TARGET == WIDTH - 1  # numeric below treats every other column after the header as a feature


@dataclass(frozen=True)
class EntityRows(TokenRows):
    @property
    def numeric(self) -> Tensor:
        # An action's target column is a row reference, not a numeric feature.
        return torch.cat((self.data[..., FEATURES:TARGET], self.data[..., TARGET:].masked_fill(
            self.is_kind(TokenKind.ACTION)[..., None], 0)), dim=-1)

    def with_targets(self, embedded: Tensor) -> Tensor:
        """Add the target monster embedding before attention; a zero reference means no target."""
        reference = self.data[..., TARGET] * self.is_kind(TokenKind.ACTION)
        index = (reference - LAYOUT.link_offset).clamp(min=0).long()
        target = embedded.gather(-2, index[..., None].expand_as(embedded))
        return embedded + target * (reference > 0)[..., None]

    def in_card_zone(self, zone: CardZone) -> Tensor:
        return self.is_kind(TokenKind.CARD) & ((self.numeric[..., 0] - zone).abs() < 0.5)
