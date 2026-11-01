"""Combat prediction contract and the two supported architectures. No deployment policy."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class ModelSpec:
    kind: str  # persisted architecture identifier; entity names PooledTransformer
    dim: int
    layers: int
    heads: int

    def __post_init__(self):
        if self.kind not in ('transformer', 'entity'):
            raise ValueError('model.kind: transformer or entity')
        if min(self.dim, self.layers, self.heads) <= 0 or self.dim % self.heads:
            raise ValueError('positive dim/layers/heads required; dim must be divisible by heads')


class CombatModel(nn.Module, ABC):
    """Tokens [B,T,68] -> win probability [B], action logits [B,T].

    Context is row zero; type zero is padding and type six is an action. Non-action
    logits are masked below -1e8. Padding must not affect real-token predictions.
    The immutable spec completely identifies the architecture, not its deployment.
    """
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec

    @abstractmethod
    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


def build_model(spec: ModelSpec) -> CombatModel:
    from .pooled_transformer import PooledTransformer
    from .transformer import Transformer

    return {'transformer': Transformer, 'entity': PooledTransformer}[spec.kind](spec)
