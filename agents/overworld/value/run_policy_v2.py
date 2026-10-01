"""Run value / card-pick network, kind "run_policy_v2" (docs/research/run-rl/network-study.md). Research stage: editable
until a variant is adopted. Same input dict and after-state scoring as run_policy_v1 (see that file), plus options:

  deck_attn  number of self-attention layers over the after-state deck (cards can see each other: synergies).
             0 = run_policy_v1's deep set (sum + mean of independently encoded cards).
  use_map    False drops the path encoder (ablation).
  heads      attention heads.
  aux        number of auxiliary outputs (logits per after-state; e.g. HP entering the boss, reached boss, floor):
             extra training signal only, never used to decide.

After-state a = deck + offered card a (skip: deck unchanged). With attention the whole deck is re-encoded per option:
tokens [B, K+1, C+1, W] (the option card is one extra token, masked out for skip), flattened to [B*(K+1), C+1, W].
Output: win_logit [B, K+1] (value: P(clear) / expected score), the only head trained by apps/run_rl.
"""
import torch
from torch import nn

ROOMS, FLOORS, NONE = 10, 15, 8
SCALARS = 6


class RunPolicyV2(nn.Module):
    KIND = "run_policy_v2"

    def __init__(self, cards=512, relics=256, potions=64, bosses=16, width=64, hidden=256, depth=2, dropout=0.1,
                 deck_attn=0, heads=4, use_map=True, aux=0):
        super().__init__()
        self.args = dict(cards=cards, relics=relics, potions=potions, bosses=bosses, width=width, hidden=hidden,
                         depth=depth, dropout=dropout, deck_attn=deck_attn, heads=heads, use_map=use_map, aux=aux)
        W = width
        self.card_id = nn.Embedding(cards, W)
        self.card_mlp = nn.Sequential(nn.Linear(2 * W + 4, W), nn.ReLU(), nn.Linear(W, W))
        self.attn = nn.ModuleList(nn.TransformerEncoderLayer(W, heads, 2 * W, dropout=0.0, batch_first=True,
                                                             norm_first=True) for _ in range(deck_attn))
        self.relic_id = nn.Embedding(relics, W)
        self.relic_mlp = nn.Sequential(nn.Linear(W + 1, W), nn.ReLU(), nn.Linear(W, W))
        self.potion_id = nn.Embedding(potions, W)
        self.boss = nn.Embedding(bosses, W)
        g = 5 * W + SCALARS + 1
        self.use_map = use_map
        if use_map:
            self.room = nn.Embedding(ROOMS, W); self.floor = nn.Embedding(FLOORS, W)
            self.path_mlp = nn.Sequential(nn.Linear(W, W), nn.ReLU(), nn.Linear(W, W))
            self.ctx = nn.Linear(g, W)
            # score_mlp(cat(path, ctx)) with its first linear layer split by input (same function, no [.., M, 2W] concat)
            self.score_path = nn.Linear(W, W)
            self.score_ctx = nn.Linear(W, W, bias=False)
            self.score_out = nn.Linear(W, 1)
        f = g + (2 * W + 1 if use_map else 0)
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(f, hidden), nn.ReLU())
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, hidden),
                                                  nn.ReLU(), nn.Linear(hidden, hidden)) for _ in range(depth))
        self.win_out = nn.Linear(hidden, 1)
        self.aux_out = nn.Linear(hidden, aux) if aux else None

    def _cards(self, ids, up, misc, ctx):
        n = ids.shape[1]
        return self.card_mlp(torch.cat([self.card_id(ids), up[..., None], misc[..., None] / 10,
                                        ctx[:, None].expand(-1, n, -1)], -1))

    def _deck(self, b, ctx0):
        """Deck vectors per after-state: dsum [B, K+1, W], size [B, K+1]."""
        cards = self._cards(b["card_id"], b["card_up"], b["card_misc"], ctx0)          # [B, C, W]
        opt = self._cards(b["opt_id"], b["opt_up"], b["opt_misc"], ctx0)               # [B, K, W]
        B, C, W = cards.shape
        K = opt.shape[1]
        cmask = b["card_mask"]
        n = cmask.sum(1, keepdim=True)
        size = torch.cat([(n + 1).expand(-1, K), n], 1)                                # [B, K+1]
        if not self.attn:
            deck = (cards * cmask[..., None]).sum(1)
            dsum = deck[:, None] + torch.cat([opt, torch.zeros_like(opt[:, :1])], 1)
            return dsum, size
        extra = torch.cat([opt, torch.zeros_like(opt[:, :1])], 1)                      # [B, K+1, W]
        tok = torch.cat([cards[:, None].expand(-1, K + 1, -1, -1), extra[:, :, None]], 2)   # [B, K+1, C+1, W]
        emask = torch.cat([torch.ones(B, K, device=cards.device), torch.zeros(B, 1, device=cards.device)], 1)
        mask = torch.cat([cmask[:, None].expand(-1, K + 1, -1), emask[..., None]], 2)       # [B, K+1, C+1]
        tok = tok.reshape(B * (K + 1), C + 1, W)
        m = mask.reshape(B * (K + 1), C + 1)
        pad = m == 0
        pad[:, 0] = pad[:, 0] & (m.sum(1) > 0)  # never fully masked (empty deck): keep token 0
        for layer in self.attn:
            tok = layer(tok, src_key_padding_mask=pad)
        dsum = (tok * m[..., None]).sum(1).reshape(B, K + 1, W)
        return dsum, size

    def forward(self, b):
        """-> (win_logit [B, K+1],)  (tuple for compatibility with run_policy_v1 callers: [0] = value logits)."""
        sc = b["scalars"]
        boss = self.boss(b["boss"])
        ctx0 = torch.cat([boss, sc[:, 2:3], sc[:, 4:5]], -1)
        dsum, size = self._deck(b, ctx0)
        K1 = dsum.shape[1]
        data = b["relic_data"]
        relic = self.relic_mlp(torch.cat([self.relic_id(b["relic_id"]),
                                          (torch.sign(data) * torch.log1p(data.abs()))[..., None]], -1))
        relic = (relic * b["relic_mask"][..., None]).sum(1)
        potion = (self.potion_id(b["potion_id"]) * b["potion_mask"][..., None]).sum(1)
        shared = torch.cat([relic, potion, boss, sc], -1)
        g = torch.cat([dsum, dsum / size[..., None].clamp_min(1), shared[:, None].expand(-1, K1, -1),
                       torch.log1p(size)[..., None]], -1)
        parts = [g]
        if self.use_map:
            room = b["path_room"]                                                        # [B, M, 15]
            Bn, M = room.shape[:2]
            # sum over floors of emb(room) + emb(floor), NONE slots excluded, without a [B, M, 15, W] tensor
            table = (self.room.weight[:, None] + self.floor.weight[None]).reshape(-1, self.room.weight.shape[1])
            idx = (room * FLOORS + torch.arange(FLOORS, device=room.device)).reshape(-1, FLOORS)
            w = (room != NONE).reshape(-1, FLOORS).float()
            summed = torch.nn.functional.embedding_bag(idx, table, mode="sum", per_sample_weights=w)
            paths = self.path_mlp(summed.reshape(Bn, M, -1))                             # [B, M, W]
            M = paths.shape[1]
            ctx = self.ctx(g)
            hid = torch.relu(self.score_path(paths)[:, None] + self.score_ctx(ctx)[:, :, None])   # [B, K+1, M, W]
            score = self.score_out(hid).squeeze(-1).masked_fill(~b["path_mask"].bool()[:, None], float("-inf"))
            best_score, best = score.max(-1)
            best_path = torch.gather(paths, 1, best.reshape(-1, K1, 1).expand(-1, -1, paths.shape[-1]))
            pm = b["path_mask"][..., None].float()
            mean_path = ((paths * pm).sum(1) / pm.sum(1).clamp_min(1))[:, None].expand(-1, K1, -1)
            parts += [best_path, best_score[..., None], mean_path]
        h = self.inp(torch.cat(parts, -1))
        for block in self.blocks:
            h = h + block(h)
        h = torch.relu(h)
        if self.aux_out is None:
            return (self.win_out(h).squeeze(-1),)
        return self.win_out(h).squeeze(-1), self.aux_out(h)                              # aux logits [B, K+1, aux]
