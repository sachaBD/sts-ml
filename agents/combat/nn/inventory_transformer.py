"""inventory_v1: explicit entities and relationships, with no pre-attention pooling."""
import torch
from torch import nn

from .model import CombatModel, context_width
from .tokens import ID_CAPACITY, MASKED_LOGIT, TokenRows

# Must match InventoryKind and the feature layouts in inventory_tokens.cpp.
CONTEXT, CARD, MONSTER, POTION, RELIC, ACTION, PLAYER, PLAYER_EFFECT, MONSTER_EFFECT, INVENTORY = range(1, 11)
FEATURE_WIDTHS = {CONTEXT: 11, CARD: 17, MONSTER: 13, POTION: 18, RELIC: 3,
                  ACTION: 60, PLAYER: 11, PLAYER_EFFECT: 4, MONSTER_EFFECT: 2, INVENTORY: 19}
OWNER, SOURCE, TARGET = 65, 66, 67
VERSION = 1


class InventoryTransformer(CombatModel):
    def __init__(self, spec):
        super().__init__(spec)
        dim = spec.dim
        self.kind = nn.Embedding(11, dim)
        self.id = nn.Embedding(11 * ID_CAPACITY, dim)
        self.id2 = nn.Embedding(ID_CAPACITY, dim)
        # Disjoint type-specific feature slices give independent projections in one matrix multiply.
        # The type embedding supplies the type-specific bias.
        self.numeric = nn.Linear(sum(FEATURE_WIDTHS.values()), dim, bias=False)
        self.owner = nn.Linear(dim, dim, bias=False)
        self.source = nn.Linear(dim, dim, bias=False)
        self.target = nn.Linear(dim, dim, bias=False)
        layer = nn.TransformerEncoderLayer(dim, spec.heads, 4 * dim, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, spec.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, context_width(spec))
        self.policy = nn.Linear(dim, 1)

    def _embed(self, tokens):
        rows = TokenRows(tokens)
        kind = rows.kind.long()
        namespace = kind
        card_source = rows.is_kind(ACTION) & ((rows.id2 == 1) | (rows.id2 == 3) | (rows.id2 == 4))
        namespace = torch.where(rows.is_kind(INVENTORY) | card_source, CARD, namespace)
        namespace = torch.where(rows.is_kind(ACTION) & (rows.id2 == 2), POTION, namespace)
        numeric = torch.cat([tokens[..., 3:3 + count] * rows.is_kind(type_)[..., None]
                             for type_, count in FEATURE_WIDTHS.items()], dim=-1)
        return (self.kind(kind) + self.id(namespace * ID_CAPACITY + rows.id)
                + self.id2(rows.id2) + self.numeric(numeric))

    def heads(self, tokens):
        rows = TokenRows(tokens)
        embedded = self._embed(tokens)
        x = embedded
        for column, projection in ((OWNER, self.owner), (SOURCE, self.source), (TARGET, self.target)):
            reference = tokens[..., column]
            index = (reference.long() - 1).clamp(min=0)
            linked = embedded.gather(1, index[..., None].expand_as(embedded))
            x = x + projection(linked) * (reference > 0)[..., None]
        hidden = self.norm(self.encoder(x, src_key_padding_mask=~rows.valid))
        context = self.value(hidden[:, 0]).float()
        # A legacy replay or incorrectly selected native encoder must not silently train this model.
        compatible = (tokens[:, 0, 0] == CONTEXT) & (tokens[:, 0, 2] == VERSION)
        context = torch.where(compatible[:, None], context, torch.full_like(context, float('nan')))
        logits = self.policy(hidden).squeeze(-1).float().masked_fill(~rows.is_kind(ACTION), MASKED_LOGIT)
        return context, logits
