"""Pooled-token transformer for Ironclad vs Champ, preserving the existing token and action contract."""
import torch
from torch import Tensor, nn

from agents.combat.nn.model import CombatModel, context_width, win_probability
from agents.combat.nn.entity.encoding import CardZone, EntityRows, KIND_CAPACITY, TokenKind
from agents.combat.nn.tokens import ID_CAPACITY, MASKED_LOGIT, ROW, WIDTH


class EntityModel(CombatModel):
    """Pool relics and draw/discard/exhaust cards, then attend over retained tokens.

    Predictions are scattered back to the original action positions.
    """
    encoding = 'entity'

    def __init__(self, spec):
        super().__init__(spec)
        dim, layers, heads = spec.dim, spec.layers, spec.heads
        self.kind = nn.Embedding(KIND_CAPACITY, dim)
        self.id = nn.Embedding(KIND_CAPACITY * ID_CAPACITY, dim)
        self.id2 = nn.Embedding(ID_CAPACITY, dim)
        self.numeric = nn.Linear(WIDTH - ROW.features, dim)
        self.item = nn.Sequential(nn.Linear(dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.group = nn.Parameter(torch.randn(4, dim) * 0.02)  # RELICS, DRAW, DISCARD, EXHAUST
        self.group_norm = nn.LayerNorm(dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, context_width(spec))
        self.policy = nn.Linear(dim, 1)

    def example_tokens(self, batch=2, length=3):
        tokens = torch.zeros(batch, length, WIDTH)
        tokens[:, 0, 0], tokens[:, 1:, 0] = TokenKind.CONTEXT, TokenKind.ACTION
        return tokens

    def heads(self, tokens):
        # Attend over the longest retained row rather than the padded length: identical predictions, and
        # ~4x fewer encoder positions on real replay (pooling leaves ~16 of ~66). Export uses predict directly.
        _, keep = self._group_masks(EntityRows(tokens))
        return self._heads(tokens, int(keep.sum(dim=1).max()))

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        # Exported for fixed shapes: every row keeps a slot, so no shape depends on the data.
        context, logits = self._heads(tokens, tokens.shape[1])
        return win_probability(context, self.spec), logits

    def _heads(self, tokens: Tensor, capacity: int) -> tuple[Tensor, Tensor]:
        rows = EntityRows(tokens)
        groups, keep = self._group_masks(rows)
        embedded = self._embed(rows)
        embedded = rows.with_targets(embedded)
        pooled = self._pool(embedded, groups)
        retained, index, valid = self._compact(embedded, keep, capacity)
        hidden = self._attend(retained, pooled, valid)
        return self._readout(rows, hidden, index)

    def _group_masks(self, rows: EntityRows) -> tuple[Tensor, Tensor]:
        """Relics, draw, discard, exhaust; every other real token stays individual."""
        groups = torch.stack([rows.is_kind(TokenKind.RELIC),
                              rows.in_card_zone(CardZone.DRAW),
                              rows.in_card_zone(CardZone.DISCARD),
                              rows.in_card_zone(CardZone.EXHAUST)], dim=1)
        return groups, rows.valid & ~groups.any(dim=1)

    def _embed(self, rows: EntityRows) -> Tensor:
        kind = rows.kind.long()
        return (self.kind(kind) + self.id(kind * ID_CAPACITY + rows.id)
                + self.id2(rows.id2) + self.numeric(rows.numeric))

    def _pool(self, embedded: Tensor, groups: Tensor) -> Tensor:
        """Shared item transform, sum per group, then normalize."""
        return self.group_norm(groups.to(embedded.dtype) @ self.item(embedded) + self.group)

    def _compact(self, embedded: Tensor, keep: Tensor, capacity: int) -> tuple[Tensor, Tensor, Tensor]:
        """Every retained row in original order, padded to `capacity` rows: embeddings, original indices, valid mask."""
        length = embedded.shape[1]
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

    def _readout(self, rows: EntityRows, hidden: Tensor, index: Tensor) -> tuple[Tensor, Tensor]:
        """Value head at context; action scores returned to wire positions."""
        value = self.value(hidden[:, 0]).float()
        pooled_count = self.group.shape[0]
        retained = torch.cat([hidden[:, :1], hidden[:, 1 + pooled_count:]], dim=1)
        scores = self.policy(retained).squeeze(-1).float()
        logits = torch.full_like(rows.kind, MASKED_LOGIT).scatter(1, index, scores)
        return value, logits.masked_fill(~rows.is_kind(TokenKind.ACTION), MASKED_LOGIT)

