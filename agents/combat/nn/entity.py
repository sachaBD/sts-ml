"""Pooled entity transformer for Ironclad vs Champ, preserving the existing token and action contract."""
import torch
from torch import nn

from .model import ACTION, WIDTH



CARD, RELIC = 2, 5  # token types (tokens.hpp); a card token's zone (CardZone: 0 hand, 1 draw, 2 discard, 3 exhaust) is column 3


class EntityNet(nn.Module):
    """docs/ENCODING_AND_TOPOLOGY.md, on today's token rows (no encoder or contract change).

    Relics and draw/discard/exhaust cards are pooled (shared item MLP, sum per group) into four tokens. The remaining
    tokens (context, Champ, hand, actions) are compacted to the front, in row order, and a standard transformer runs
    over context + the four pooled tokens + the rest. Logits are scattered back to the row's positions.

    attended: compacted slots in a fixed-shape export (None = all, exact for any row: training and the CPU model).
    A row with more than `attended` such tokens gets a NaN value, so the GPU backend re-evaluates it on the CPU.
    """
    def __init__(self, dim=192, layers=4, heads=None, attended=None):
        super().__init__()
        heads = dim // 32 if heads is None else heads
        if min(dim, layers, heads) <= 0 or dim % heads:
            raise ValueError('positive dim/layers/heads required; dim must be divisible by heads')
        self.spec = dict(kind='entity', dim=dim, layers=layers, heads=heads)
        self.attended = attended
        self.kind = nn.Embedding(8, dim)
        self.id = nn.Embedding(8 * 1024, dim)
        self.id2 = nn.Embedding(1024, dim)
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
        b, t = tokens.shape[0], tokens.shape[1]
        kind, zone = tokens[..., 0], tokens[..., 3]
        card, relic = (kind - CARD).abs() < 0.5, (kind - RELIC).abs() < 0.5
        piles = [card & ((zone - z).abs() < 0.5) for z in (1, 2, 3)]
        keep = (kind > 0.5) & ~relic & ~piles[0] & ~piles[1] & ~piles[2]
        x = (self.kind(kind.long()) + self.id(kind.long() * 1024 + tokens[..., 1].long()) + self.id2(tokens[..., 2].long())
             + self.numeric(tokens[..., 3:]))
        groups = torch.stack([relic] + piles, 1).to(x.dtype)                      # [B, 4, T]
        pooled = self.group_norm(groups @ self.item(x) + self.group)              # [B, 4, D]
        a = t if self.attended is None else min(self.attended, t)
        # Kept tokens first, in row order (the context token is position 0, so it stays first).
        order = keep.float() * (2 * t) - torch.arange(t, device=tokens.device, dtype=torch.float32)
        index = order.topk(a, dim=1).indices                                      # [B, A]
        kept = keep.gather(1, index)
        x = x.gather(1, index[..., None].expand(-1, -1, x.shape[-1]))
        pad = torch.cat([~kept[:, :1], torch.zeros(b, 4, dtype=torch.bool, device=tokens.device), ~kept[:, 1:]], 1)
        h = self.norm(self.encoder(torch.cat([x[:, :1], pooled, x[:, 1:]], 1), src_key_padding_mask=pad))
        value = torch.sigmoid(self.value(h[:, 0]).squeeze(-1).float())
        value = torch.where(keep.sum(1) > a, torch.full_like(value, float('nan')), value)
        scores = self.policy(torch.cat([h[:, :1], h[:, 5:]], 1)).squeeze(-1).float()  # [B, A]
        logits = torch.full((b, t), -1e9, device=tokens.device).scatter(1, index, scores)
        return value, logits.masked_fill((kind - ACTION).abs() > 0.5, -1e9)
