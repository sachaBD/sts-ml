"""Combat prediction contract and persisted architectures. No deployment policy."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from agents.combat.nn.layout import Layout


# Outcome bins on one margin axis: losses by enemy HP left (most first), then wins by player HP (least first).
LOSS_BINS = WIN_BINS = 10
# Training-only predictions after the value logits when spec.auxiliary; replay.TARGETS names.
AUXILIARY = ('hp_lost', 'enemy_hp_lost', 'turns_remaining')


@dataclass(frozen=True)
class ModelSpec:
    kind: str  # persisted architecture identifier; entity names EntityModel
    dim: int
    layers: int
    heads: int
    value: str  # win: one win logit; outcome: LOSS_BINS + WIN_BINS outcome logits
    auxiliary: bool  # AUXILIARY predictions follow the value logits
    encoding_version: int = field(default=2, init=False)  # action-to-target row references

    def __post_init__(self):
        if self.kind not in ('entity', 'inventory_v1'):
            raise ValueError('model.kind: entity or inventory_v1')
        if self.value not in ('win', 'outcome'):
            raise ValueError('model.value: win or outcome')
        if min(self.dim, self.layers, self.heads) <= 0 or self.dim % self.heads:
            raise ValueError('positive dim/layers/heads required; dim must be divisible by heads')


def value_width(spec: ModelSpec) -> int:
    return 1 if spec.value == 'win' else LOSS_BINS + WIN_BINS


def context_width(spec: ModelSpec) -> int:
    return value_width(spec) + len(AUXILIARY) * spec.auxiliary


def win_probability(context: Tensor, spec: ModelSpec) -> Tensor:
    """Context readout [B, context_width] -> [B]; a NaN row stays NaN."""
    value = context.float()[:, :value_width(spec)]
    if spec.value == 'win':
        return torch.sigmoid(value[:, 0])
    # Summed softmax mass can round above 1 in fp32 or fp16; NaNs must still propagate.
    return value.softmax(-1)[:, LOSS_BINS:].sum(-1).clamp(0, 1)


class CombatModel(nn.Module, ABC):
    """Tokens [B,T,68] -> win probability [B], action logits [B,T]. Training reads the value head itself.

    Context is row zero; type zero is padding and type six is an action. Non-action
    logits are masked below -1e8. Padding must not affect real-token predictions.
    The immutable spec completely identifies the architecture, not its deployment.
    """
    layout: Layout  # its encoding's layout.toml, which the native encoder writes

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec

    @abstractmethod
    def example_tokens(self, batch=2, length=3) -> Tensor:
        """A CPU input in this model's encoding for tracing and warm-up: a context row, then `length - 1` legal
        actions."""
        raise NotImplementedError

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        context, logits = self.heads(tokens)
        return win_probability(context, self.spec), logits

    @abstractmethod
    def heads(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Context readout [B, context_width(spec)] (value logits, then AUXILIARY) and action logits [B,T]."""
        raise NotImplementedError


def build_model(spec: ModelSpec) -> CombatModel:
    from agents.combat.nn.entity.model import EntityModel
    from agents.combat.nn.inventory.model import InventoryModel

    return {'entity': EntityModel, 'inventory_v1': InventoryModel}[spec.kind](spec)
