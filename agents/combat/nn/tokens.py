"""Typed view of the numeric token rows every encoding writes (tokens.hpp Row): kind, id and id2, then features.

Tensors deliberately remain numeric for PyTorch/ONNX. Each encoding's layout is its layout.toml (layout.py).
"""
from dataclasses import dataclass

from torch import Tensor

MASKED_LOGIT = -1e9
FEATURES = 3  # the first feature column, after kind, id and id2 (layout.toml header)


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

    def is_kind(self, kind: int) -> Tensor:
        # Float comparisons avoid ONNX CUDA's unsupported integer Equal path.
        return (self.kind - kind).abs() < 0.5
