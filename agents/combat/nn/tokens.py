"""Typed view of the numeric token rows every encoding writes (tokens.cpp): kind, id and id2, then features.

Tensors deliberately remain numeric for PyTorch/ONNX. Each encoding's layout is in layout.toml.
"""
from dataclasses import dataclass

from torch import Tensor

from agents.combat.nn.layout import LAYOUT

WIDTH, ROW = LAYOUT.width, LAYOUT.row
ID_CAPACITY = 1024  # id embeddings per kind, part of each model's architecture
MASKED_LOGIT = -1e9


@dataclass(frozen=True)
class TokenRows:
    """Named field access; the underlying numeric wire tensor is never changed."""
    data: Tensor

    @property
    def kind(self) -> Tensor:
        return self.data[..., ROW.kind]

    @property
    def id(self) -> Tensor:
        return self.data[..., ROW.id].long()

    @property
    def id2(self) -> Tensor:
        return self.data[..., ROW.id2].long()

    @property
    def valid(self) -> Tensor:
        return self.kind > 0.5

    def is_kind(self, kind: int) -> Tensor:
        # Float comparisons avoid ONNX CUDA's unsupported integer Equal path.
        return (self.kind - kind).abs() < 0.5
