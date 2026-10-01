"""Pre-combat outcome model (kind "combat_outcome_v2"). Frozen once used: see topology/README.md.

Same inputs, outputs, bins and per-encounter linear term as combat_outcome_v1 (drop-in for train_marginals.py).
Differences, aimed at card-in-context effects (a card's value depends on the fight and the rest of the deck):
    card   = card_mlp(emb(card_id), upgraded, misc / 10, emb(encounter), hp / max_hp)   encounter-conditioned
    deck   = cat(sum(card), mean(card))        sum keeps count information, mean keeps composition
    f      = cat(deck, sum(relic), sum(potion), emb(encounter), scalars)
    h      = `depth` residual MLP blocks of width `hidden` (depth >= 1)
"""
import torch
from torch import nn

from sts_combat_rl.topology.combat_outcome_v1 import BINS


class CombatOutcomeV2(nn.Module):
    KIND = "combat_outcome_v2"

    def __init__(self, cards=512, relics=256, potions=64, encounters=64, width=32, hidden=64, depth=2, dropout=0.0):
        super().__init__()
        self.args = dict(cards=cards, relics=relics, potions=potions, encounters=encounters, width=width, hidden=hidden,
                         depth=depth, dropout=dropout)
        self.card_id = nn.Embedding(cards, width)
        self.card_mlp = nn.Sequential(nn.Linear(2 * width + 3, width), nn.ReLU(), nn.Linear(width, width))
        self.relic_id = nn.Embedding(relics, width)
        self.relic_mlp = nn.Sequential(nn.Linear(width + 1, width), nn.ReLU(), nn.Linear(width, width))
        self.potion_id = nn.Embedding(potions, width)
        self.encounter = nn.Embedding(encounters, width)
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(5 * width + 5, hidden), nn.ReLU())
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                                  nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(depth))
        self.win_out = nn.Linear(hidden, 1)
        self.hp_out = nn.Linear(hidden, BINS)
        self.enc_linear = nn.Parameter(torch.zeros(encounters, 2, 1 + BINS))  # (intercept, slope) x (win, bins)

    def forward(self, b):
        enc = self.encounter(b["encounter"])
        mask = b["card_mask"][..., None]
        n = b["card_id"].shape[1]
        ctx = torch.cat([enc, b["scalars"][:, 2:3]], -1)[:, None].expand(-1, n, -1)
        card = self.card_mlp(torch.cat([self.card_id(b["card_id"]), b["card_up"][..., None],
                                        b["card_misc"][..., None] / 10, ctx], -1)) * mask
        deck = torch.cat([card.sum(1), card.sum(1) / mask.sum(1).clamp_min(1)], -1)
        data = b["relic_data"]
        relic = self.relic_mlp(torch.cat([self.relic_id(b["relic_id"]),
                                          (torch.sign(data) * torch.log1p(data.abs()))[..., None]], -1))
        relic = (relic * b["relic_mask"][..., None]).sum(1)
        potion = (self.potion_id(b["potion_id"]) * b["potion_mask"][..., None]).sum(1)
        h = self.inp(torch.cat([deck, relic, potion, enc, b["scalars"]], -1))
        for block in self.blocks:
            h = h + block(h)
        h = torch.relu(h)
        coef = self.enc_linear[b["encounter"]]
        linear = coef[:, 0] + coef[:, 1] * b["scalars"][:, :1]
        return self.win_out(h).squeeze(-1) + linear[:, 0], self.hp_out(h) + linear[:, 1:]
