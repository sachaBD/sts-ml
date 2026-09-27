"""Deep Sets value net v1 (kind "deep_sets_v1") for v3 public-state encodings. Frozen: see topology/README.md."""

import torch
from torch import nn


class DeepSetsV1(nn.Module):
    """Deep Sets value net v1: context-free token MLPs, summed per zone / monsters / interactions, one
    hidden head layer, tanh value."""

    KIND = "deep_sets_v1"

    def __init__(self, card_vocab, monster_vocab, move_vocab, width):
        super().__init__()
        self.config = {
            "kind": self.KIND,
            "card_vocab": card_vocab,
            "monster_vocab": monster_vocab,
            "move_vocab": move_vocab,
            "width": width,
        }
        self.input_state = nn.Embedding(64, 4)
        self.card_selection_task = nn.Embedding(32, 4)
        self.card_id = nn.Embedding(card_vocab, 8)
        self.zone = nn.Embedding(5, 4)
        self.card_type = nn.Embedding(5, 3)
        self.target_type = nn.Embedding(4, 3)
        self.card_mlp = nn.Sequential(
            nn.Linear(32, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
        )
        self.monster_id = nn.Embedding(monster_vocab, 8)
        self.move = nn.Embedding(move_vocab, 8)
        self.monster_mlp = nn.Sequential(
            nn.Linear(25, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
        )
        self.interaction_mlp = nn.Sequential(
            nn.Linear(2 * width + 6, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(50 + 8 + 6 * width, width),
            nn.ReLU(),
            nn.Linear(width, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _sum(tokens, owners, batch_size):
        result = tokens.new_zeros((batch_size, tokens.shape[-1]))
        if tokens.numel():
            result.index_add_(0, owners, tokens)
        return result

    def forward(
        self,
        global_numeric,
        input_state,
        card_selection_task,
        card_ids,
        card_zones,
        card_types,
        target_types,
        card_numeric,
        monster_ids,
        move_ids,
        monster_numeric,
        interaction_cards,
        interaction_monsters,
        interaction_numeric,
        card_state_indices,
        monster_state_indices,
        interaction_state_indices,
    ):
        batch_size = global_numeric.shape[0]
        card = self.card_mlp(
            torch.cat(
                (
                    self.card_id(card_ids),
                    self.zone(card_zones),
                    self.card_type(card_types),
                    self.target_type(target_types),
                    card_numeric,
                ),
                -1,
            )
        )
        pools = [
            self._sum(
                card[card_zones == zone],
                card_state_indices[card_zones == zone],
                batch_size,
            )
            for zone in range(4)
        ]
        monster = self.monster_mlp(
            torch.cat(
                (self.monster_id(monster_ids), self.move(move_ids), monster_numeric), -1
            )
        )
        if len(interaction_cards):
            interaction = self.interaction_mlp(
                torch.cat(
                    (
                        card[interaction_cards],
                        monster[interaction_monsters],
                        interaction_numeric,
                    ),
                    -1,
                )
            )
        else:
            interaction = card.new_zeros((0, card.shape[-1]))
        features = torch.cat(
            (
                global_numeric,
                self.input_state(input_state),
                self.card_selection_task(card_selection_task),
                *pools,
                self._sum(monster, monster_state_indices, batch_size),
                self._sum(interaction, interaction_state_indices, batch_size),
            ),
            -1,
        )
        return self.head(features).squeeze(-1)
