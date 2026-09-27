"""Deep Sets value net v2 (kind "deep_sets_v2") for v3 public-state encodings. Frozen: see topology/README.md."""

import torch
from torch import nn


class _HeadBlock(nn.Module):
    """Pre-LayerNorm residual block: h + fc2(relu(fc1(norm(h))))."""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, h):
        return h + self.fc2(torch.relu(self.fc1(self.norm(h))))


class DeepSetsV2(nn.Module):
    """Deep Sets value net v2 (architecture kind "deep_sets_v2"): v1's context-free token encoders (wider),
    pooled per zone / monsters / interactions, then a normalised residual head.

        f = cat(global_numeric, emb(input_state), emb(card_selection_task), 6 pools [, log1p(6 counts)])
        f = head_in_norm(f)                               # if head_input_norm
        h = relu(head_in(f))
        for block in head_blocks: h = h + block.fc2(relu(block.fc1(block.norm(h))))
        h = head_out_norm(h)                              # if head_blocks > 0
        value = sigmoid|tanh(value_out(h)); won_logit = won_out(h); hp = sigmoid(hp_out(h))   # aux if aux_heads

    Pools (and counts): card zones 0-3 (hand, draw, discard, exhaust), monsters, interactions.
    Parameter names are read by topology/value_net.cpp: input_state, card_selection_task, card_id, zone,
    card_type, target_type, card_mlp.{0,2}, monster_id, move, monster_mlp.{0,2}, interaction_mlp.{0,2},
    head_in_norm, head_in, head_blocks.{i}.{norm,fc1,fc2}, head_out_norm, value_out, won_out, hp_out.
    """

    KIND = "deep_sets_v2"
    GLOBAL_NUMERIC = 50
    CARD_NUMERIC = 14
    MONSTER_NUMERIC = 9
    INTERACTION_NUMERIC = 6
    POOLS = 6

    def __init__(self, card_vocab, monster_vocab, move_vocab, width, card_id_dim, monster_id_dim, move_dim,
                 pool_count_features, head_input_norm, head_width, head_blocks, output, aux_heads, zero_init_blocks):
        super().__init__()
        if output not in ("sigmoid", "tanh"):
            raise ValueError(f"output {output!r} is not 'sigmoid' or 'tanh'")
        self.config = {
            "kind": self.KIND, "card_vocab": card_vocab, "monster_vocab": monster_vocab, "move_vocab": move_vocab,
            "width": width, "card_id_dim": card_id_dim, "monster_id_dim": monster_id_dim, "move_dim": move_dim,
            "pool_count_features": bool(pool_count_features), "head_input_norm": bool(head_input_norm),
            "head_width": head_width, "head_blocks": head_blocks, "output": output, "aux_heads": bool(aux_heads),
            "zero_init_blocks": bool(zero_init_blocks),
        }
        self.pool_count_features = bool(pool_count_features)
        self.output = output
        self.aux_heads = bool(aux_heads)
        self.input_state = nn.Embedding(64, 4)
        self.card_selection_task = nn.Embedding(32, 4)
        self.card_id = nn.Embedding(card_vocab, card_id_dim)
        self.zone = nn.Embedding(5, 4)
        self.card_type = nn.Embedding(5, 3)
        self.target_type = nn.Embedding(4, 3)
        self.card_mlp = nn.Sequential(
            nn.Linear(card_id_dim + 4 + 3 + 3 + self.CARD_NUMERIC, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU())
        self.monster_id = nn.Embedding(monster_vocab, monster_id_dim)
        self.move = nn.Embedding(move_vocab, move_dim)
        self.monster_mlp = nn.Sequential(
            nn.Linear(monster_id_dim + move_dim + self.MONSTER_NUMERIC, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU())
        self.interaction_mlp = nn.Sequential(
            nn.Linear(2 * width + self.INTERACTION_NUMERIC, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU())
        features = self.GLOBAL_NUMERIC + 4 + 4 + self.POOLS * width + (self.POOLS if self.pool_count_features else 0)
        self.head_in_norm = nn.LayerNorm(features) if head_input_norm else None
        self.head_in = nn.Linear(features, head_width)
        self.head_blocks = nn.ModuleList(_HeadBlock(head_width) for _ in range(head_blocks))
        if zero_init_blocks:  # each residual block starts as the identity
            for block in self.head_blocks:
                nn.init.zeros_(block.fc2.weight)
                nn.init.zeros_(block.fc2.bias)
        self.head_out_norm = nn.LayerNorm(head_width) if head_blocks > 0 else None
        self.value_out = nn.Linear(head_width, 1)
        if self.aux_heads:
            self.won_out = nn.Linear(head_width, 1)
            self.hp_out = nn.Linear(head_width, 1)

    @staticmethod
    def _sum(tokens, owners, batch_size):
        result = tokens.new_zeros((batch_size, tokens.shape[-1]))
        if tokens.numel():
            result.index_add_(0, owners, tokens)
        return result

    def trunk(self, global_numeric, input_state, card_selection_task, card_ids, card_zones, card_types, target_types,
              card_numeric, monster_ids, move_ids, monster_numeric, interaction_cards, interaction_monsters,
              interaction_numeric, card_state_indices, monster_state_indices, interaction_state_indices):
        """The head's last hidden layer h [B, head_width]."""
        batch_size = global_numeric.shape[0]
        card = self.card_mlp(torch.cat((self.card_id(card_ids), self.zone(card_zones), self.card_type(card_types),
                                        self.target_type(target_types), card_numeric), -1))
        monster = self.monster_mlp(torch.cat((self.monster_id(monster_ids), self.move(move_ids), monster_numeric), -1))
        if len(interaction_cards):
            interaction = self.interaction_mlp(torch.cat(
                (card[interaction_cards], monster[interaction_monsters], interaction_numeric), -1))
        else:
            interaction = card.new_zeros((0, card.shape[-1]))
        owners = [card_state_indices[card_zones == zone] for zone in range(4)]
        pools = [self._sum(card[card_zones == zone], owners[zone], batch_size) for zone in range(4)]
        pools.append(self._sum(monster, monster_state_indices, batch_size))
        pools.append(self._sum(interaction, interaction_state_indices, batch_size))
        parts = [global_numeric, self.input_state(input_state), self.card_selection_task(card_selection_task), *pools]
        if self.pool_count_features:
            counts = global_numeric.new_zeros((batch_size, self.POOLS))
            for column, index in enumerate((*owners, monster_state_indices, interaction_state_indices)):
                if index.numel():
                    counts[:, column].index_add_(0, index, torch.ones_like(index, dtype=counts.dtype))
            parts.append(torch.log1p(counts))
        f = torch.cat(parts, -1)
        if self.head_in_norm is not None:
            f = self.head_in_norm(f)
        h = torch.relu(self.head_in(f))
        for block in self.head_blocks:
            h = block(h)
        if self.head_out_norm is not None:
            h = self.head_out_norm(h)
        return h

    def _value(self, h):
        v = self.value_out(h).squeeze(-1)
        return torch.sigmoid(v) if self.output == "sigmoid" else torch.tanh(v)

    def forward(self, *args, **kwargs):
        return self._value(self.trunk(*args, **kwargs))

    def forward_all(self, *args, **kwargs):
        """{"value": [B], "won_logit": [B] (raw logit), "hp": [B] (sigmoid)}; aux keys only with aux_heads."""
        h = self.trunk(*args, **kwargs)
        out = {"value": self._value(h)}
        if self.aux_heads:
            out["won_logit"] = self.won_out(h).squeeze(-1)
            out["hp"] = torch.sigmoid(self.hp_out(h)).squeeze(-1)
        return out
