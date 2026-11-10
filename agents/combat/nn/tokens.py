"""Typed view of the numeric token rows both encodings write (tokens.cpp): kind, id and id2, then features.

Tensors deliberately remain numeric for PyTorch/ONNX. Each model's own layout lives with it
(entity/encoding.py, inventory/model.py).
"""
from dataclasses import dataclass

from torch import Tensor

WIDTH = 68
ID_CAPACITY = 1024
MASKED_LOGIT = -1e9


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
