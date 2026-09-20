import torch
from torch import nn


class DeepSetsValue(nn.Module):
    def __init__(self, card_vocab: int = 512, monster_vocab: int = 128, move_vocab: int = 512, width: int = 32):
        super().__init__()
        self.input_state = nn.Embedding(64, 4)
        self.card_selection_task = nn.Embedding(32, 4)
        self.card_id = nn.Embedding(card_vocab, 8)
        self.zone = nn.Embedding(5, 4)
        self.card_type = nn.Embedding(5, 3)
        self.target_type = nn.Embedding(4, 3)
        self.card_mlp = nn.Sequential(nn.Linear(32, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.monster_id = nn.Embedding(monster_vocab, 8)
        self.move = nn.Embedding(move_vocab, 8)
        self.monster_mlp = nn.Sequential(nn.Linear(25, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.interaction_mlp = nn.Sequential(nn.Linear(2 * width + 6, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(22 + 8 + 6 * width, width), nn.ReLU(), nn.Linear(width, 1), nn.Tanh())

    def forward(
        self,
        global_numeric: torch.Tensor,
        input_state: torch.Tensor,
        card_selection_task: torch.Tensor,
        card_ids: torch.Tensor,
        card_zones: torch.Tensor,
        card_types: torch.Tensor,
        target_types: torch.Tensor,
        card_numeric: torch.Tensor,
        monster_ids: torch.Tensor,
        move_ids: torch.Tensor,
        monster_numeric: torch.Tensor,
        interaction_cards: torch.Tensor,
        interaction_monsters: torch.Tensor,
        interaction_numeric: torch.Tensor,
    ) -> torch.Tensor:
        card = self.card_mlp(
            torch.cat((self.card_id(card_ids), self.zone(card_zones), self.card_type(card_types), self.target_type(target_types), card_numeric), dim=-1)
        )
        card_pools = [card[card_zones == zone].sum(dim=0) for zone in range(4)]
        monster = self.monster_mlp(torch.cat((self.monster_id(monster_ids), self.move(move_ids), monster_numeric), dim=-1))
        interaction = self.interaction_mlp(torch.cat((card[interaction_cards], monster[interaction_monsters], interaction_numeric), dim=-1))
        return self.head(
            torch.cat((
                global_numeric,
                self.input_state(input_state),
                self.card_selection_task(card_selection_task),
                *card_pools,
                monster.sum(dim=0),
                interaction.sum(dim=0),
            )).unsqueeze(0)
        ).squeeze(0)
