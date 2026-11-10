"""inventory_v1: explicit entities and relationships, with no pre-attention pooling."""
import torch
from torch import nn

from agents.combat.nn.layout import LAYOUT
from agents.combat.nn.model import CombatModel, context_width
from agents.combat.nn.tokens import ID_CAPACITY, MASKED_LOGIT, ROW, WIDTH, TokenRows

INVENTORY_V1 = LAYOUT.inventory_v1  # layout.toml's [inventory_v1], which inventory/encoding.cpp writes
_kind = INVENTORY_V1.kind
CONTEXT, CARD, MONSTER, POTION, RELIC, ACTION, PLAYER, PLAYER_EFFECT, MONSTER_EFFECT, INVENTORY = (
    k.id for k in (_kind.context, _kind.card, _kind.monster, _kind.potion, _kind.relic, _kind.action, _kind.player,
                   _kind.player_effect, _kind.monster_effect, _kind.inventory))
FEATURE_WIDTHS = {k.id: k.features for k in sorted(vars(_kind).values(), key=lambda k: k.id)}
OWNER, SOURCE, TARGET = INVENTORY_V1.link.owner, INVENTORY_V1.link.source, INVENTORY_V1.link.target
VERSION = INVENTORY_V1.version
_action = INVENTORY_V1.action_kind
CARD_ACTION, POTION_ACTION, SINGLE_CARD_SELECTION, MULTI_CARD_SELECTION = (
    _action.card, _action.potion, _action.single_card_selection, _action.multi_card_selection)


class InventoryModel(CombatModel):
    encoding = 'inventory_v1'

    def __init__(self, spec):
        super().__init__(spec)
        dim = spec.dim
        self.kind = nn.Embedding(11, dim)
        self.id = nn.Embedding(11 * ID_CAPACITY, dim)
        self.id2 = nn.Embedding(ID_CAPACITY, dim)
        # Disjoint type-specific feature slices give independent projections in one matrix multiply.
        # The type embedding supplies the type-specific bias.
        self.numeric = nn.Linear(sum(FEATURE_WIDTHS.values()), dim, bias=False)
        self.effects = nn.Linear(dim, dim, bias=False)
        self.owner = nn.Linear(dim, dim, bias=False)
        self.source = nn.Linear(dim, dim, bias=False)
        self.target = nn.Linear(dim, dim, bias=False)
        layer = nn.TransformerEncoderLayer(dim, spec.heads, 4 * dim, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, spec.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, context_width(spec))
        self.policy = nn.Linear(dim, 1)

    def example_tokens(self, batch=2, length=3):
        tokens = torch.zeros(batch, length, WIDTH)
        tokens[:, 0, 0], tokens[:, 0, 2], tokens[:, 1:, 0] = CONTEXT, VERSION, ACTION
        return tokens

    def _embed(self, tokens):
        rows = TokenRows(tokens)
        kind = rows.kind.long()
        action_kind = tokens[..., 2]
        card_source = rows.is_kind(ACTION) & (((action_kind - CARD_ACTION).abs() < .5)
                                             | ((action_kind - SINGLE_CARD_SELECTION).abs() < .5)
                                             | ((action_kind - MULTI_CARD_SELECTION).abs() < .5))
        potion_source = rows.is_kind(ACTION) & ((action_kind - POTION_ACTION).abs() < .5)
        namespace = torch.where(rows.is_kind(INVENTORY) | card_source, CARD, kind)
        namespace = torch.where(potion_source, POTION, namespace)
        numeric = torch.cat([tokens[..., ROW.features:ROW.features + count] * rows.is_kind(type_)[..., None]
                             for type_, count in FEATURE_WIDTHS.items()], dim=-1)
        return (self.kind(kind) + self.id(namespace * ID_CAPACITY + rows.id)
                + self.id2(rows.id2) + self.numeric(numeric))

    def heads(self, tokens):
        rows = TokenRows(tokens)
        embedded = self._embed(tokens)
        # Attach effects to their owners before actions read their sources/targets. Otherwise two
        # identical monsters with different debuffs would still have identical target embeddings.
        positions = torch.arange(tokens.shape[1], device=tokens.device, dtype=tokens.dtype) + 1
        ownership = (tokens[..., OWNER, None] - positions).abs() < .5
        entities = embedded + ownership.transpose(1, 2).to(embedded.dtype) @ self.effects(embedded)
        x = entities
        for column, projection in ((OWNER, self.owner), (SOURCE, self.source), (TARGET, self.target)):
            reference = tokens[..., column]
            index = (reference.long() - INVENTORY_V1.link_offset).clamp(min=0)
            linked = entities.gather(1, index[..., None].expand_as(entities))
            x = x + projection(linked) * (reference > 0)[..., None]
        hidden = self.norm(self.encoder(x, src_key_padding_mask=~rows.valid))
        context = self.value(hidden[:, 0]).float()
        # An entity replay or incorrectly selected native encoder must not silently train this model.
        compatible = (tokens[:, 0, 0] == CONTEXT) & (tokens[:, 0, 2] == VERSION)
        context = torch.where(compatible[:, None], context, torch.full_like(context, float('nan')))
        logits = self.policy(hidden).squeeze(-1).float().masked_fill(~rows.is_kind(ACTION), MASKED_LOGIT)
        return context, logits
