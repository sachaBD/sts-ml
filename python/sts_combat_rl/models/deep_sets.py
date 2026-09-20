"""Permutation-invariant value network for v2 public-state encodings."""

import torch
from torch import nn


class DeepSetsValue(nn.Module):
    def __init__(self, card_vocab=512, monster_vocab=128, move_vocab=512, width=64):
        super().__init__()
        self.config = {
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
            nn.Linear(22 + 8 + 6 * width, width),
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
        card_state_indices=None,
        monster_state_indices=None,
        interaction_state_indices=None,
    ):
        if global_numeric.dim() == 1:
            global_numeric = global_numeric.unsqueeze(0)
            input_state = input_state.reshape(1)
            card_selection_task = card_selection_task.reshape(1)
        batch_size = global_numeric.shape[0]
        device = global_numeric.device
        if card_state_indices is None:
            card_state_indices = torch.zeros(
                len(card_ids), dtype=torch.long, device=device
            )
        if monster_state_indices is None:
            monster_state_indices = torch.zeros(
                len(monster_ids), dtype=torch.long, device=device
            )
        if interaction_state_indices is None:
            interaction_state_indices = torch.zeros(
                len(interaction_cards), dtype=torch.long, device=device
            )
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
