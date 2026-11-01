"""Typed view of the numeric token wire format written by tokens.cpp.

Enums describe the schema; tensors deliberately remain numeric for PyTorch/ONNX.
CardZone applies only to card tokens: other token kinds reuse the numeric columns.
"""
from dataclasses import dataclass
from enum import IntEnum

from torch import Tensor

WIDTH = 68
KIND_CAPACITY = 8
ID_CAPACITY = 1024
MASKED_LOGIT = -1e9


class TokenKind(IntEnum):
    PADDING = 0
    CONTEXT = 1
    CARD = 2
    MONSTER = 3
    POTION = 4
    RELIC = 5
    ACTION = 6


class CardZone(IntEnum):
    HAND = 0
    DRAW = 1
    DISCARD = 2
    EXHAUST = 3
    OFFERED = 4


@dataclass(frozen=True)
class TokenRows:
    """Named field access; the underlying numeric wire tensor is never changed."""
    data: Tensor

    @property
    def kind(self) -> Tensor:
        return self.data[..., 0]

    @property
    def id(self) -> Tensor:
        return self.data[..., 1].long()

    @property
    def id2(self) -> Tensor:
        return self.data[..., 2].long()

    @property
    def numeric(self) -> Tensor:
        return self.data[..., 3:]

    @property
    def valid(self) -> Tensor:
        return self.kind > 0.5

    def is_kind(self, kind: TokenKind) -> Tensor:
        # Float comparisons avoid ONNX CUDA's unsupported integer Equal path.
        return (self.kind - kind).abs() < 0.5

    def in_card_zone(self, zone: CardZone) -> Tensor:
        return self.is_kind(TokenKind.CARD) & ((self.numeric[..., 0] - zone).abs() < 0.5)
