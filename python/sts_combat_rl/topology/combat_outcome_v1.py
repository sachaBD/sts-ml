"""Pre-combat outcome model (kind "combat_outcome_v1"). Frozen once used: see topology/README.md.

Input: the persistent state right before a fight (combat_transition_v1 `pre`) plus the encounter:
    card   = card_mlp(emb(card_id), upgraded, misc / 10)          per deck card
    relic  = relic_mlp(emb(relic_id), symlog(data))                per relic
    potion = emb(potion_id)                                        per held potion
    f = cat(sum(card), sum(relic), sum(potion), emb(encounter), hp / 100, max_hp / 100, hp / max_hp,
            potion_capacity / 5, log1p(deck size))
    h = MLP(f)
    out = linear(h) + per-encounter (intercept + slope * hp / 100) terms (the baseline's form, so the set
          encoders learn a residual on it)
Output (the endpoint is the in-battle HP before exitBattle, combat_v3 `final_hp`):
    win_logit          P(win) = sigmoid
    hp_logits[BINS]    P(HP bin | win): bins 1-5, 6-10, ..., 96-100, then one overflow bin > 100
"""
import torch
from torch import nn

HP_BIN = 5
HP_CAP = 100
BINS = HP_CAP // HP_BIN + 1  # + overflow


def hp_bin(hp: torch.Tensor) -> torch.Tensor:
    """Final HP (>= 1, on a win) -> bin index; > HP_CAP -> the overflow bin."""
    return torch.clamp((hp - 1) // HP_BIN, 0, BINS - 1)


class CombatOutcomeV1(nn.Module):
    KIND = "combat_outcome_v1"

    def __init__(self, cards=512, relics=256, potions=64, encounters=64, width=64, hidden=128, dropout=0.0):
        super().__init__()
        self.args = dict(cards=cards, relics=relics, potions=potions, encounters=encounters, width=width, hidden=hidden,
                         dropout=dropout)
        self.card_id = nn.Embedding(cards, width)
        self.card_mlp = nn.Sequential(nn.Linear(width + 2, width), nn.ReLU(), nn.Linear(width, width))
        self.relic_id = nn.Embedding(relics, width)
        self.relic_mlp = nn.Sequential(nn.Linear(width + 1, width), nn.ReLU(), nn.Linear(width, width))
        self.potion_id = nn.Embedding(potions, width)
        self.encounter = nn.Embedding(encounters, width)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width + 5, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.win_out = nn.Linear(hidden, 1)
        self.hp_out = nn.Linear(hidden, BINS)
        self.enc_linear = nn.Parameter(torch.zeros(encounters, 2, 1 + BINS))  # (intercept, slope) x (win, bins)

    def forward(self, b):
        """b: dict of padded tensors (card_id / card_up / card_misc / card_mask [B, C], relic_id / relic_data /
        relic_mask [B, R], potion_id / potion_mask [B, P], encounter [B], scalars [B, 5])."""
        card = self.card_mlp(torch.cat([self.card_id(b["card_id"]), b["card_up"][..., None],
                                        b["card_misc"][..., None] / 10], -1))
        card = (card * b["card_mask"][..., None]).sum(1)
        data = b["relic_data"]
        relic = self.relic_mlp(torch.cat([self.relic_id(b["relic_id"]),
                                          (torch.sign(data) * torch.log1p(data.abs()))[..., None]], -1))
        relic = (relic * b["relic_mask"][..., None]).sum(1)
        potion = (self.potion_id(b["potion_id"]) * b["potion_mask"][..., None]).sum(1)
        h = self.head(torch.cat([card, relic, potion, self.encounter(b["encounter"]), b["scalars"]], -1))
        coef = self.enc_linear[b["encounter"]]
        linear = coef[:, 0] + coef[:, 1] * b["scalars"][:, :1]
        return self.win_out(h).squeeze(-1) + linear[:, 0], self.hp_out(h) + linear[:, 1:]
