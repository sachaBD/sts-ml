"""Typed view of the shared numeric token wire format written by tokens.cpp.

Tensors deliberately remain numeric for PyTorch/ONNX. Each model's own layout lives with it
(entity/encoding.py, inventory/model.py); the kinds below are shared by both.
"""
from dataclasses import dataclass
from enum import IntEnum

from torch import Tensor

WIDTH = 68
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
    def valid(self) -> Tensor:
        return self.kind > 0.5

    def is_kind(self, kind: TokenKind) -> Tensor:
        # Float comparisons avoid ONNX CUDA's unsupported integer Equal path.
        return (self.kind - kind).abs() < 0.5
