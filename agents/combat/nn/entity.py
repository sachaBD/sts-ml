"""Pooled entity transformer for Ironclad vs Champ, preserving the existing token and action contract."""
import torch
from torch import Tensor, nn

from .model import CombatModel
from .tokens import CardZone, ID_CAPACITY, KIND_CAPACITY, MASKED_LOGIT, TokenKind, TokenRows, WIDTH


class EntityNet(CombatModel):
    """Pool relics and draw/discard/exhaust cards, then attend over retained tokens.

    Predictions are scattered back to the original action positions. A bounded view
    can be requested by an export adapter; overflow is explicit rather than silently
    dropping actions. Normal forward evaluates every retained token.
    """
    def __init__(self, spec):
        super().__init__(spec)
        dim, layers, heads = spec.dim, spec.layers, spec.heads
        self.kind = nn.Embedding(KIND_CAPACITY, dim)
        self.id = nn.Embedding(KIND_CAPACITY * ID_CAPACITY, dim)
        self.id2 = nn.Embedding(ID_CAPACITY, dim)
        self.numeric = nn.Linear(WIDTH - 3, dim)
        self.item = nn.Sequential(nn.Linear(dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.group = nn.Parameter(torch.randn(4, dim) * 0.02)  # RELICS, DRAW, DISCARD, EXHAUST
        self.group_norm = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, 1)
        self.policy = nn.Linear(dim, 1)

    def forward(self, tokens):
        return self.predict(tokens)

    def predict(self, tokens: Tensor, slots: int | None = None) -> tuple[Tensor, Tensor]:
        """Evaluate an optional bounded attention view; overflow returns a NaN value."""
        rows = TokenRows(tokens)
        groups, keep = self._group_masks(rows)
        embedded = self._embed(rows)
        pooled = self._pool(embedded, groups)
        retained, index, valid = self._compact(embedded, keep, slots)
        hidden = self._attend(retained, pooled, valid)
        return self._readout(rows, hidden, index, keep)

    def _group_masks(self, rows: TokenRows) -> tuple[Tensor, Tensor]:
        """Relics, draw, discard, exhaust; every other real token stays individual."""
        groups = torch.stack([rows.is_kind(TokenKind.RELIC),
                              rows.in_card_zone(CardZone.DRAW),
                              rows.in_card_zone(CardZone.DISCARD),
                              rows.in_card_zone(CardZone.EXHAUST)], dim=1)
        return groups, rows.valid & ~groups.any(dim=1)

    def _embed(self, rows: TokenRows) -> Tensor:
        kind = rows.kind.long()
        return (self.kind(kind) + self.id(kind * ID_CAPACITY + rows.id)
                + self.id2(rows.id2) + self.numeric(rows.numeric))

    def _pool(self, embedded: Tensor, groups: Tensor) -> Tensor:
        """Shared item transform, sum per group, then normalize."""
        return self.group_norm(groups.to(embedded.dtype) @ self.item(embedded) + self.group)

    def _compact(self, embedded: Tensor, keep: Tensor, slots: int | None) -> tuple[Tensor, Tensor, Tensor]:
        """Retained embeddings, original indices and valid mask, in original row order."""
        length = embedded.shape[1]
        capacity = length if slots is None else min(slots, length)
        order = keep.float() * (2 * length) - torch.arange(length, device=embedded.device, dtype=torch.float32)
        index = order.topk(capacity, dim=1).indices
        retained = embedded.gather(1, index[..., None].expand(-1, -1, embedded.shape[-1]))
        return retained, index, keep.gather(1, index)

    def _attend(self, retained: Tensor, pooled: Tensor, valid: Tensor) -> Tensor:
        """Insert pooled groups after context; padded compacted slots never participate."""
        pooled_padding = torch.zeros_like(pooled[..., 0], dtype=torch.bool)
        padding = torch.cat([~valid[:, :1], pooled_padding, ~valid[:, 1:]], dim=1)
        sequence = torch.cat([retained[:, :1], pooled, retained[:, 1:]], dim=1)
        return self.norm(self.encoder(sequence, src_key_padding_mask=padding))

    def _readout(self, rows: TokenRows, hidden: Tensor, index: Tensor, keep: Tensor) -> tuple[Tensor, Tensor]:
        """Value at context; action scores returned to wire positions, with explicit overflow."""
        value = torch.sigmoid(self.value(hidden[:, 0]).squeeze(-1).float())
        overflow = keep.sum(dim=1) > index.shape[1]
        value = torch.where(overflow, torch.full_like(value, float('nan')), value)
        pooled_count = self.group.shape[0]
        retained = torch.cat([hidden[:, :1], hidden[:, 1 + pooled_count:]], dim=1)
        scores = self.policy(retained).squeeze(-1).float()
        logits = torch.full_like(rows.kind, MASKED_LOGIT).scatter(1, index, scores)
        return value, logits.masked_fill(~rows.is_kind(TokenKind.ACTION), MASKED_LOGIT)
