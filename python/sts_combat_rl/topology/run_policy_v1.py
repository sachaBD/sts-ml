"""Card-pick policy + run value model (kind "run_policy_v1"). Frozen once trained: see topology/README.md and
slop_docs/card-policy-loop/README.md (agents 3 + 4 as one network). Research stage: untrained, still editable.

A card reward is a deck edit, so every option is scored as an AFTER-STATE (deck + card; skip = the deck unchanged).
The deck pool is additive, so after-states cost one card vector each:
    card_i  = card_mlp(emb(card_id), upgraded, misc / 10, ctx0)     ctx0 = emb(boss), hp / max_hp, floor / 17
    deck(a) = cat(sum_i card_i + card_a, (sum_i card_i + card_a) / (n + 1))      (skip: card_a = 0, n unchanged)
    relic   = sum relic_mlp(emb(relic_id), symlog(data));  potion = sum emb(potion_id)
    g(a)    = cat(deck(a), relic, potion, emb(boss), scalars, log1p(deck size after a))

Map = the set of remaining PATHS (enumerated by the encoder: every route from the current node to the boss, deduplicated
by room sequence). Each path is a room sequence indexed by absolute floor (NONE = floor already behind us):
    path_p  = path_mlp(sum_floors (emb(room) + emb(floor)))                              what this route contains
    score_p = score_mlp(path_p, ctx(a))      ctx(a) = linear(g(a))                       how good this route is for
                                                                                         this deck / HP (after-state a)
    map(a)  = cat(path_{p*}, score_{p*}, mean_p path_p),   p* = argmax_p score_p         max pooling over whole paths:
                                                                                         the best route is taken as a unit
    f(a)    = cat(g(a), map(a))  ->  `depth` residual MLP blocks (width `hidden`)  ->  heads
The route score gets gradient through score_{p*} (hard max, like max pooling); later it can be supervised directly with
"P(clear) if I take this route", which also gives a path-choice head for free (first step of the best route).

Heads, per option a (K candidate cards, then skip as the last option):
    win_logit[a]      value: P(clear the act | take a, then play on with the current policy) = sigmoid
    hp_logits[a, :]   value: P(HP after the act boss bin | clear), bins as combat_outcome_v1 (1-5, ..., 96-100, >100)
    policy_logit[a]   policy: probabilities = softmax over valid options (invalid options masked to -inf)
Also returned: best_path [B, K+1] (index of p* per option). The state value V(s) is the skip option's value head.

Input dict (B = batch, C/R/P = padded deck/relic/potion sizes, K = max offered cards, M = max paths):
    card_id, card_up, card_misc, card_mask [B, C];  relic_id, relic_data, relic_mask [B, R];  potion_id, potion_mask [B, P]
    opt_id, opt_up, opt_misc, opt_mask [B, K]       offered cards (mask 0 = empty slot); skip is implicit
    boss [B]                                        act boss id
    scalars [B, 6]                                  hp/100, max_hp/100, hp/max_hp, gold/100, floor/17, potion_capacity/5
    path_room [B, M, 15]                            sts::Room per floor for each remaining path (8 = NONE: behind us)
    path_mask [B, M]                                1 = real path
"""
import torch
from torch import nn

from sts_combat_rl.topology.combat_outcome_v1 import BINS

ROOMS, FLOORS, NONE = 10, 15, 8
SCALARS = 6


class RunPolicyV1(nn.Module):
    KIND = "run_policy_v1"

    def __init__(self, cards=512, relics=256, potions=64, bosses=16, width=32, hidden=64, depth=1, dropout=0.3):
        super().__init__()
        self.args = dict(cards=cards, relics=relics, potions=potions, bosses=bosses, width=width, hidden=hidden,
                         depth=depth, dropout=dropout)
        self.card_id = nn.Embedding(cards, width)
        self.card_mlp = nn.Sequential(nn.Linear(2 * width + 4, width), nn.ReLU(), nn.Linear(width, width))
        self.relic_id = nn.Embedding(relics, width)
        self.relic_mlp = nn.Sequential(nn.Linear(width + 1, width), nn.ReLU(), nn.Linear(width, width))
        self.potion_id = nn.Embedding(potions, width)
        self.boss = nn.Embedding(bosses, width)
        self.room = nn.Embedding(ROOMS, width); self.floor = nn.Embedding(FLOORS, width)
        self.path_mlp = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        g = 5 * width + SCALARS + 1
        self.ctx = nn.Linear(g, width)
        self.score_mlp = nn.Sequential(nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 1))
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(g + 2 * width + 1, hidden), nn.ReLU())
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                                  nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(depth))
        self.win_out = nn.Linear(hidden, 1)
        self.hp_out = nn.Linear(hidden, BINS)
        self.policy_out = nn.Linear(hidden, 1)

    def _cards(self, ids, up, misc, ctx):
        n = ids.shape[1]
        return self.card_mlp(torch.cat([self.card_id(ids), up[..., None], misc[..., None] / 10,
                                        ctx[:, None].expand(-1, n, -1)], -1))

    def _paths(self, b):
        """Encode each remaining path independently of the deck: [B, M, W]."""
        room = b["path_room"]
        tok = (self.room(room) + self.floor.weight[None, None]) * (room != NONE)[..., None]
        return self.path_mlp(tok.sum(2))

    def forward(self, b):
        """-> win_logit [B, K+1], hp_logits [B, K+1, BINS], policy_logit [B, K+1] (masked), best_path [B, K+1]."""
        sc = b["scalars"]
        boss = self.boss(b["boss"])
        ctx0 = torch.cat([boss, sc[:, 2:3], sc[:, 4:5]], -1)
        deck = (self._cards(b["card_id"], b["card_up"], b["card_misc"], ctx0) * b["card_mask"][..., None]).sum(1)
        n = b["card_mask"].sum(1, keepdim=True)
        opt = self._cards(b["opt_id"], b["opt_up"], b["opt_misc"], ctx0)           # [B, K, W]
        K = opt.shape[1]
        dsum = deck[:, None] + torch.cat([opt, torch.zeros_like(opt[:, :1])], 1)   # skip adds nothing
        size = torch.cat([(n + 1).expand(-1, K), n], 1)                              # [B, K+1]
        data = b["relic_data"]
        relic = self.relic_mlp(torch.cat([self.relic_id(b["relic_id"]),
                                          (torch.sign(data) * torch.log1p(data.abs()))[..., None]], -1))
        relic = (relic * b["relic_mask"][..., None]).sum(1)
        potion = (self.potion_id(b["potion_id"]) * b["potion_mask"][..., None]).sum(1)
        shared = torch.cat([relic, potion, boss, sc], -1)
        g = torch.cat([dsum, dsum / size[..., None].clamp_min(1), shared[:, None].expand(-1, K + 1, -1),
                       torch.log1p(size)[..., None]], -1)                           # [B, K+1, G]
        # Map: score every path for every after-state, max-pool over whole paths.
        paths = self._paths(b)                                                       # [B, M, W]
        M = paths.shape[1]
        ctx = self.ctx(g)                                                            # [B, K+1, W]
        pair = torch.cat([paths[:, None].expand(-1, K + 1, -1, -1), ctx[:, :, None].expand(-1, -1, M, -1)], -1)
        score = self.score_mlp(pair).squeeze(-1)                                     # [B, K+1, M]
        score = score.masked_fill(~b["path_mask"].bool()[:, None], float("-inf"))
        best_score, best = score.max(-1)                                             # [B, K+1]
        best_path = torch.gather(paths, 1, best.reshape(-1, (K + 1), 1).expand(-1, -1, paths.shape[-1]))
        pm = b["path_mask"][..., None].float()
        mean_path = ((paths * pm).sum(1) / pm.sum(1).clamp_min(1))[:, None].expand(-1, K + 1, -1)
        h = self.inp(torch.cat([g, best_path, best_score[..., None], mean_path], -1))
        for block in self.blocks:
            h = h + block(h)
        h = torch.relu(h)
        valid = torch.cat([b["opt_mask"].bool(), torch.ones_like(b["opt_mask"][:, :1]).bool()], 1)
        policy = self.policy_out(h).squeeze(-1).masked_fill(~valid, float("-inf"))
        return self.win_out(h).squeeze(-1), self.hp_out(h), policy, best

    @torch.no_grad()
    def choose(self, b, mode="sample", temperature=1.0):
        """Pick an option per row. mode: "sample" (from policy probabilities; exploration / self-play),
        "greedy" (argmax policy; evaluation), "value" (argmax after-state P(clear); search-free baseline).
        Returns (option index [B], probabilities [B, K+1]); index K = skip."""
        win, _, logit, _ = self(b)
        probs = torch.softmax(logit / temperature, -1)
        if mode == "sample":
            return torch.multinomial(probs, 1).squeeze(-1), probs
        if mode == "greedy":
            return probs.argmax(-1), probs
        if mode == "value":
            return win.masked_fill(logit.isinf(), float("-inf")).argmax(-1), probs
        raise ValueError(mode)
