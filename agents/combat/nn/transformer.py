"""Full-token combat transformer."""
import torch
from torch import nn

from .model import CombatModel
from .tokens import ID_CAPACITY, KIND_CAPACITY, MASKED_LOGIT, TokenKind, TokenRows, WIDTH


class Transformer(CombatModel):
    def __init__(self, spec):
        super().__init__(spec)
        dim, layers, heads = spec.dim, spec.layers, spec.heads
        self.kind = nn.Embedding(KIND_CAPACITY, dim)
        self.id = nn.Embedding(KIND_CAPACITY * ID_CAPACITY, dim)   # per type: type * 1024 + id
        self.id2 = nn.Embedding(ID_CAPACITY, dim)
        self.numeric = nn.Linear(WIDTH - 3, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, 1)
        self.policy = nn.Linear(dim, 1)

    def forward(self, tokens):
        rows = TokenRows(tokens)
        kind = rows.kind.long()
        x = (self.kind(kind) + self.id(kind * ID_CAPACITY + rows.id)
             + self.id2(rows.id2) + self.numeric(rows.numeric))
        h = self.norm(self.encoder(x, src_key_padding_mask=~rows.valid))
        value = torch.sigmoid(self.value(h[:, 0]).squeeze(-1).float())
        logits = self.policy(h).squeeze(-1).float().masked_fill(~rows.is_kind(TokenKind.ACTION), MASKED_LOGIT)
        return value, logits
