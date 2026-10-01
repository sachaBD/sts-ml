"""Deep Sets value net v3 (kind "deep_sets_v3") for v4 encodings. Frozen once used: see agents/combat/value/README.md."""

import torch
from torch import nn

from .deep_sets_v2 import _HeadBlock


class DeepSetsV3(nn.Module):
    """Deep Sets value net v3 (architecture kind "deep_sets_v3"): deep_sets_v2's trunk on the v4 encoding
    (environments/combat/encoding_v4.cpp: player statuses, monster statuses and previous move, potion and relic tokens),
    with the terminal score formula built into the output.

        card    = card_mlp(card_id, zone, card_type, target_type, card numeric)                  (as v2)
        monster = monster_mlp(monster_id, move, previous move, monster numeric, monster status)
        inter   = interaction_mlp(card, monster, interaction numeric)                            (as v2)
        potion  = potion_mlp(potion_id, potion numeric)
        relic   = relic_mlp(relic_id, relic numeric)
        f = cat(global_numeric, player_numeric, emb(input_state), emb(card_selection_task), 8 pools
                [, log1p(8 counts)])
        h = v2's head (head_in_norm?, head_in, residual head_blocks, head_out_norm)

    Pools (and counts): card zones 0-3 (hand, draw, discard, exhaust), monsters, interactions, potions, relics.

    Output: the expected terminal value of combat_v3 (agents/combat/search/teacher_search.cpp outcome_columns),
        terminal = won * (score_hp_offset + final_hp + score_potion_hp * final_potions)
                       / (score_max_hp_offset + max_hp),
    factored into three heads, each conditional on winning so that the product is exact:
        p_win = sigmoid(won_out(h))
        hp    = sigmoid(hp_out(h)) * max_hp          E[final HP | win]
        kept  = sigmoid(keep_out(h)) * potions       E[potions left | win]; potions = held now (potion tokens)
        value = p_win * (score_hp_offset + hp + score_potion_hp * kept) / (score_max_hp_offset + max_hp)
    max_hp is the current max HP (global max_hp); the terminal uses the fight's starting max HP, the same
    unless max HP changed mid-fight (Fruit Juice).
    Training: value against the target as before; won_logit with BCE against won; on won rows hp_fraction
    against final_hp / max_hp and, where potions > 0, keep_fraction against final_potions / potions.

    Policy head (policy_width > 0): one logit per legal action, AlphaZero-style priors for the search. An action
    token (environments/combat/encoding.hpp ActionToken: kind, card-selection task, and the optional source card, target
    monster, potion and card x target interaction) is encoded with the trunk's own token encoders:
        a = cat(emb(kind), emb(task), card, monster, potion, interaction,
                [skips_selection, discards_potion, 4 presence flags])
            (absent parts are zero)
        logit = policy_out(relu(policy_state(h)[owner] + policy_action(a)))
    policy_action has no bias and depends on the action token only, so C++ caches it per token. Training: cross
    entropy against the root visit distribution, identical tokens of a state merged (data.py).

    id_dropout: in training mode, each potion / relic id is replaced by the unknown id 0 with this
    probability, so rare ids lean on their numeric description. No effect in eval mode (or in C++).

    Parameter names are read by agents/combat/value/value_net.cpp: v2's (card_mlp, monster_mlp, interaction_mlp,
    embeddings, head_in_norm, head_in, head_blocks, head_out_norm) plus potion_id, potion_mlp.{0,2},
    relic_id, relic_mlp.{0,2}, won_out, hp_out, keep_out, and with the policy head action_kind, policy_state,
    policy_action (no bias), policy_out.
    """

    KIND = "deep_sets_v3"
    ENCODING_VERSION = 4
    # widths of environments/combat/encoding.hpp (v4)
    GLOBAL_NUMERIC = 50
    PLAYER_NUMERIC = 12
    CARD_NUMERIC = 14
    MONSTER_NUMERIC = 9
    MONSTER_STATUS = 15
    INTERACTION_NUMERIC = 6
    POTION_NUMERIC = 18
    RELIC_NUMERIC = 3
    POOLS = 8
    ACTION_KINDS = 8  # EncodedActionKind (5 used)
    ACTION_FLAGS = 6

    def __init__(self, card_vocab, monster_vocab, move_vocab, potion_vocab, relic_vocab, width, card_id_dim,
                 monster_id_dim, move_dim, potion_id_dim, relic_id_dim, id_dropout, pool_count_features,
                 head_input_norm, head_width, head_blocks, zero_init_blocks, score_hp_offset, score_potion_hp,
                 score_max_hp_offset, policy_width):
        super().__init__()
        self.config = {
            "kind": self.KIND, "card_vocab": card_vocab, "monster_vocab": monster_vocab, "move_vocab": move_vocab,
            "potion_vocab": potion_vocab, "relic_vocab": relic_vocab, "width": width, "card_id_dim": card_id_dim,
            "monster_id_dim": monster_id_dim, "move_dim": move_dim, "potion_id_dim": potion_id_dim,
            "relic_id_dim": relic_id_dim, "id_dropout": float(id_dropout),
            "pool_count_features": bool(pool_count_features), "head_input_norm": bool(head_input_norm),
            "head_width": head_width, "head_blocks": head_blocks, "zero_init_blocks": bool(zero_init_blocks),
            "score_hp_offset": float(score_hp_offset), "score_potion_hp": float(score_potion_hp),
            "score_max_hp_offset": float(score_max_hp_offset), "policy_width": int(policy_width),
        }
        self.pool_count_features = bool(pool_count_features)
        self.id_dropout = float(id_dropout)
        self.score = (float(score_hp_offset), float(score_potion_hp), float(score_max_hp_offset))
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
        self.move = nn.Embedding(move_vocab, move_dim)  # current and previous move share it
        self.monster_mlp = nn.Sequential(
            nn.Linear(monster_id_dim + 2 * move_dim + self.MONSTER_NUMERIC + self.MONSTER_STATUS, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU())
        self.interaction_mlp = nn.Sequential(
            nn.Linear(2 * width + self.INTERACTION_NUMERIC, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU())
        self.potion_id = nn.Embedding(potion_vocab, potion_id_dim)
        self.potion_mlp = nn.Sequential(
            nn.Linear(potion_id_dim + self.POTION_NUMERIC, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU())
        self.relic_id = nn.Embedding(relic_vocab, relic_id_dim)
        self.relic_mlp = nn.Sequential(
            nn.Linear(relic_id_dim + self.RELIC_NUMERIC, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU())
        features = (self.GLOBAL_NUMERIC + self.PLAYER_NUMERIC + 4 + 4 + self.POOLS * width
                    + (self.POOLS if self.pool_count_features else 0))
        self.head_in_norm = nn.LayerNorm(features) if head_input_norm else None
        self.head_in = nn.Linear(features, head_width)
        self.head_blocks = nn.ModuleList(_HeadBlock(head_width) for _ in range(head_blocks))
        if zero_init_blocks:  # each residual block starts as the identity
            for block in self.head_blocks:
                nn.init.zeros_(block.fc2.weight)
                nn.init.zeros_(block.fc2.bias)
        self.head_out_norm = nn.LayerNorm(head_width) if head_blocks > 0 else None
        self.won_out = nn.Linear(head_width, 1)
        self.hp_out = nn.Linear(head_width, 1)
        self.keep_out = nn.Linear(head_width, 1)
        self.policy_width = int(policy_width)
        if self.policy_width > 0:
            self.action_kind = nn.Embedding(self.ACTION_KINDS, 4)
            self.policy_state = nn.Linear(head_width, self.policy_width)
            self.policy_action = nn.Linear(4 + 4 + 4 * width + self.ACTION_FLAGS, self.policy_width, bias=False)
            self.policy_out = nn.Linear(self.policy_width, 1)

    @staticmethod
    def _sum(tokens, owners, batch_size):
        result = tokens.new_zeros((batch_size, tokens.shape[-1]))
        if tokens.numel():
            result.index_add_(0, owners, tokens)
        return result

    def _drop_ids(self, ids):
        if not self.training or self.id_dropout <= 0 or ids.numel() == 0:
            return ids
        return torch.where(torch.rand(ids.shape, device=ids.device) < self.id_dropout, torch.zeros_like(ids), ids)

    def trunk(self, global_numeric, player_numeric, input_state, card_selection_task, card_ids, card_zones,
              card_types, target_types, card_numeric, monster_ids, move_ids, previous_move_ids, monster_numeric,
              monster_status, interaction_cards, interaction_monsters, interaction_numeric, potion_ids,
              potion_numeric, relic_ids, relic_numeric, card_state_indices, monster_state_indices,
              interaction_state_indices, potion_state_indices, relic_state_indices):
        """(h [B, head_width], potion counts [B])."""
        batch_size = global_numeric.shape[0]
        card = self.card_mlp(torch.cat((self.card_id(card_ids), self.zone(card_zones), self.card_type(card_types),
                                        self.target_type(target_types), card_numeric), -1))
        monster = self.monster_mlp(torch.cat((self.monster_id(monster_ids), self.move(move_ids),
                                              self.move(previous_move_ids), monster_numeric, monster_status), -1))
        if len(interaction_cards):
            interaction = self.interaction_mlp(torch.cat(
                (card[interaction_cards], monster[interaction_monsters], interaction_numeric), -1))
        else:
            interaction = card.new_zeros((0, card.shape[-1]))
        potion = self.potion_mlp(torch.cat((self.potion_id(self._drop_ids(potion_ids)), potion_numeric), -1))
        relic = self.relic_mlp(torch.cat((self.relic_id(self._drop_ids(relic_ids)), relic_numeric), -1))
        owners = [card_state_indices[card_zones == zone] for zone in range(4)]
        owners += [monster_state_indices, interaction_state_indices, potion_state_indices, relic_state_indices]
        pools = [self._sum(card[card_zones == zone], owners[zone], batch_size) for zone in range(4)]
        pools += [self._sum(tokens, owners[4 + i], batch_size) for i, tokens in enumerate((monster, interaction,
                                                                                           potion, relic))]
        counts = global_numeric.new_zeros((batch_size, self.POOLS))
        for column, index in enumerate(owners):
            if index.numel():
                counts[:, column].index_add_(0, index, torch.ones_like(index, dtype=counts.dtype))
        parts = [global_numeric, player_numeric, self.input_state(input_state),
                 self.card_selection_task(card_selection_task), *pools]
        if self.pool_count_features:
            parts.append(torch.log1p(counts))
        f = torch.cat(parts, -1)
        if self.head_in_norm is not None:
            f = self.head_in_norm(f)
        h = torch.relu(self.head_in(f))
        for block in self.head_blocks:
            h = block(h)
        if self.head_out_norm is not None:
            h = self.head_out_norm(h)
        return h, counts[:, 6]

    def policy_logits(self, h, action_state_indices, action_kinds, action_tasks, action_skips, action_discards,
                      action_card_mask,
                      action_card_ids, action_card_zones, action_card_types, action_target_types, action_card_numeric,
                      action_monster_mask, action_monster_ids, action_move_ids, action_previous_move_ids,
                      action_monster_numeric, action_monster_status, action_potion_mask, action_potion_ids,
                      action_potion_numeric, action_interaction_mask, action_interaction_numeric):
        """One logit per action token [A]; action_state_indices: the state (row of h) of each action."""
        card = self.card_mlp(torch.cat((self.card_id(action_card_ids), self.zone(action_card_zones),
                                        self.card_type(action_card_types), self.target_type(action_target_types),
                                        action_card_numeric), -1)) * action_card_mask[:, None]
        monster = self.monster_mlp(torch.cat((self.monster_id(action_monster_ids), self.move(action_move_ids),
                                              self.move(action_previous_move_ids), action_monster_numeric,
                                              action_monster_status), -1)) * action_monster_mask[:, None]
        potion = self.potion_mlp(torch.cat((self.potion_id(self._drop_ids(action_potion_ids)), action_potion_numeric),
                                           -1)) * action_potion_mask[:, None]
        interaction = self.interaction_mlp(torch.cat((card, monster, action_interaction_numeric), -1)) \
            * action_interaction_mask[:, None]
        flags = torch.stack((action_skips, action_discards, action_card_mask, action_monster_mask,
                             action_potion_mask, action_interaction_mask), -1)
        a = torch.cat((self.action_kind(action_kinds), self.card_selection_task(action_tasks), card, monster, potion,
                       interaction, flags), -1)
        hidden = torch.relu(self.policy_state(h)[action_state_indices] + self.policy_action(a))
        return self.policy_out(hidden).squeeze(-1)

    def forward_all(self, *args, max_hp, **kwargs):
        """{"value", "won_logit", "hp_fraction" (E[final HP | win] / max_hp; also as "hp"), "keep_fraction"
        (E[potions left | win] / potions held)}, each [B], and "policy_logits" [A] if action_* inputs are given
        (policy head only). max_hp: [B], global max_hp."""
        actions = {k: kwargs.pop(k) for k in list(kwargs) if k.startswith("action_")}
        h, potions = self.trunk(*args, **kwargs)
        won_logit = self.won_out(h).squeeze(-1)
        hp_fraction = torch.sigmoid(self.hp_out(h)).squeeze(-1)
        keep_fraction = torch.sigmoid(self.keep_out(h)).squeeze(-1)
        hp_offset, potion_hp, max_hp_offset = self.score
        value = torch.sigmoid(won_logit) * (hp_offset + hp_fraction * max_hp + potion_hp * keep_fraction * potions) \
            / (max_hp_offset + max_hp)
        out = {"value": value, "won_logit": won_logit, "hp_fraction": hp_fraction, "hp": hp_fraction,
               "keep_fraction": keep_fraction}
        if actions and self.policy_width > 0:
            out["policy_logits"] = self.policy_logits(h, **actions)
        return out

    def forward(self, *args, **kwargs):
        return self.forward_all(*args, **kwargs)["value"]
