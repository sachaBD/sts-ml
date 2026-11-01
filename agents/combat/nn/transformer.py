"""Full-token combat transformer."""
import torch
from torch import nn

from .model import ACTION, WIDTH, CombatModel


class Transformer(CombatModel):
    def __init__(self, spec):
        super().__init__(spec)
        dim, layers, heads = spec.dim, spec.layers, spec.heads
        self.kind = nn.Embedding(8, dim)
        self.id = nn.Embedding(8 * 1024, dim)   # per type: type * 1024 + id
        self.id2 = nn.Embedding(1024, dim)
        self.numeric = nn.Linear(WIDTH - 3, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, 1)
        self.policy = nn.Linear(dim, 1)

    def forward(self, tokens):
        kind = tokens[..., 0].long()
        x = (self.kind(kind) + self.id(kind * 1024 + tokens[..., 1].long()) + self.id2(tokens[..., 2].long())
             + self.numeric(tokens[..., 3:]))
        # Masks use < / > on the float type column: ONNX Runtime's CUDA provider runs no Equal here (CPU fallback).
        h = self.norm(self.encoder(x, src_key_padding_mask=tokens[..., 0] < 0.5))
        value = torch.sigmoid(self.value(h[:, 0]).squeeze(-1).float())
        logits = self.policy(h).squeeze(-1).float().masked_fill((tokens[..., 0] - ACTION).abs() > 0.5, -1e9)
        return value, logits
