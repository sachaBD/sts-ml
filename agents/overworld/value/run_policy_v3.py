"""State-conditioned backward map DAG + after-state value network.
v3.1: independent card/relic tokens with sum+mean pooling.
v3.2: the same network plus card/relic/global self-attention before pooling.
Input version: graph_encoding.py; no hidden RNG or pre-rolled future schedules.
Model output[0] stays [B, K+1], skip last, for existing policy/learner compatibility.
"""
import torch
from torch import nn

from .graph_encoding import CARD_FEATURES, NUMERIC, ROWS


def mlp(n, w, out=None):
    return nn.Sequential(nn.Linear(n,w), nn.GELU(), nn.Linear(w,out or w))


def pool(tokens, mask):
    count = mask.sum(-1, keepdim=True)
    total = (tokens * mask[...,None]).sum(-2)
    return total, total / count.clamp_min(1), torch.log1p(count)


def alternatives(tokens, mask, score):
    """Best, mean, learned soft choice, log count; all-masked sets yield zeros."""
    valid = mask.bool()
    safe_score = score.masked_fill(~valid, -1e9)
    weights = torch.softmax(safe_score, -1) * mask
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
    soft = (tokens * weights[...,None]).sum(-2)
    count = mask.sum(-1, keepdim=True)
    mean = (tokens * mask[...,None]).sum(-2) / count.clamp_min(1)
    best_index = safe_score.argmax(-1)
    best = tokens.gather(-2, best_index[...,None,None].expand(*best_index.shape,1,tokens.shape[-1])).squeeze(-2)
    best = best * (count > 0)
    return torch.cat([best,mean,soft,torch.log1p(count)],-1)


class RunPolicyV3(nn.Module):
    KIND = 'run_policy_v3_1'

    def __init__(self, cards=512, relics=256, potions=64, bosses=16, width=64, hidden=128, depth=2,
                 dropout=0.1, heads=4):
        super().__init__()
        self.args = dict(cards=cards,relics=relics,potions=potions,bosses=bosses,width=width,hidden=hidden,
                         depth=depth,dropout=dropout,heads=heads)
        if width % heads:
            raise ValueError('width must be divisible by heads')
        W = width
        self.card_id = nn.Embedding(cards,W)
        self.relic_id = nn.Embedding(relics,W)
        self.potion_id = nn.Embedding(potions,W)
        self.boss = nn.Embedding(bosses,W)
        self.act = nn.Embedding(5,W)
        self.room = nn.Embedding(10,W)
        self.floor = nn.Embedding(ROWS,W)
        self.encounter = nn.Embedding(64,W)
        self.event = nn.Embedding(64,W)
        self.event_kind = nn.Embedding(3,W)
        self.relic_kind = nn.Embedding(5,W)
        self.event_mlp = mlp(2*W+1,W)
        self.candidate_mlp = mlp(2*W+2,W)
        # boss, act, room, potion pool, 2 encounter distributions, 4 history/visible embeddings,
        # event sum+mean and candidate relic sum+mean = 14 W, plus 2 pool counts.
        self.global_mlp = mlp(14*W+len(NUMERIC)+2,W)
        self.card_mlp = mlp(2*W+2+CARD_FEATURES,W)
        self.relic_mlp = mlp(2*W+1,W)
        self.token_type = nn.Embedding(3,W)
        self.attention = nn.ModuleList() # v3.2 overrides this only
        self.context_mlp = mlp(5*W+2,W) # global + card/relic sum & mean + their counts
        self.node_mlp = mlp(3*W+3,W) # room + floor + after-state context + numerical node features
        self.branch_score = mlp(2*W,W,1)
        self.node_update = mlp(4*W+1,W)
        self.node_norm = nn.LayerNorm(W)
        self.head_in = nn.Sequential(nn.Linear(4*W+1,hidden),nn.GELU())
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU(),
                                                  nn.Dropout(dropout),nn.Linear(hidden,hidden)) for _ in range(depth))
        self.head_norm = nn.LayerNorm(hidden)
        self.win_out = nn.Linear(hidden,1)

    def _global(self,b):
        potion = (self.potion_id(b['potion_id'])*b['potion_mask'][...,None]).sum(1)
        hallway = b['hallway_probability'] @ self.encounter.weight
        elite = b['elite_probability'] @ self.encounter.weight
        history = self.encounter(b['encounter_history']).flatten(1)
        ev = self.event_mlp(torch.cat([self.event(b['event_id']), self.event_kind(b['event_kind']),
                                      b['event_eligible'][...,None]],-1))
        es,em,ec = pool(ev,b['event_mask'])
        rc = self.candidate_mlp(torch.cat([self.relic_id(b['candidate_relic_id']),
                                         self.relic_kind(b['candidate_relic_kind']),
                                         b['candidate_relic_eligible'][...,None],
                                         b['candidate_relic_shop_eligible'][...,None]],-1))
        rs,rm,rn = pool(rc,b['candidate_relic_mask'])
        return self.global_mlp(torch.cat([self.boss(b['boss']),self.act(b['act']),self.room(b['current_room']),
                                          potion,hallway,elite,history,es,em,rs,rm,b['run_numeric'],ec,rn],-1))

    def _context(self,b):
        g = self._global(b)
        B,C = b['card_id'].shape
        K = b['opt_id'].shape[1]; A = K+1; W = g.shape[-1]
        def cards(ids,up,misc,mechanics):
            return self.card_mlp(torch.cat([self.card_id(ids),up[...,None],misc[...,None]/10,mechanics,
                                            g[:,None].expand(-1,ids.shape[1],-1)],-1)) + self.token_type.weight[0]
        deck = cards(b['card_id'],b['card_up'],b['card_misc'],b['card_mechanics'])
        opt = cards(b['opt_id'],b['opt_up'],b['opt_misc'],b['opt_mechanics'])
        extra = torch.cat([opt,torch.zeros_like(opt[:,:1])],1)
        ct = torch.cat([deck[:,None].expand(-1,A,-1,-1),extra[:,:,None]],2)
        cm = torch.cat([b['card_mask'][:,None].expand(-1,A,-1),
                       torch.cat([b['opt_mask'],torch.zeros_like(b['opt_mask'][:,:1])],1)[...,None]],2)
        data = b['relic_data']; data = data.sign()*data.abs().log1p()
        rt = self.relic_mlp(torch.cat([self.relic_id(b['relic_id']),data[...,None],
                                      g[:,None].expand(-1,data.shape[1],-1)],-1)) + self.token_type.weight[1]
        rt = rt[:,None].expand(-1,A,-1,-1)
        rmask = b['relic_mask'][:,None].expand(-1,A,-1)
        if self.attention:
            tok = torch.cat([ct,rt,(g+self.token_type.weight[2])[:,None,None].expand(-1,A,1,-1)],2)
            mask = torch.cat([cm,rmask,torch.ones(B,A,1,device=g.device)],2)
            flat = tok.reshape(B*A,-1,W)
            for layer in self.attention:
                flat = layer(flat,src_key_padding_mask=~mask.reshape(B*A,-1).bool())
            tok = flat.reshape(B,A,-1,W)
            ct,rt,ga = tok[:,:,:C+1],tok[:,:,C+1:-1],tok[:,:,-1]
        else:
            ga = g[:,None].expand(-1,A,-1)
        cs,ca,cn = pool(ct,cm); rs,ra,rn = pool(rt,rmask)
        return self.context_mlp(torch.cat([ga,cs,ca,rs,ra,cn,rn],-1))

    def map_features(self,b,ctx):
        """Backward dynamic program over 16 rows, shared suffixes computed once per after-state.
        Returns root summaries [B,A,3W+1] and per-node scores [B,A,16,7].
        This is learned message passing, NOT an exact HP/deck transition simulation.
        """
        B,A,W = ctx.shape
        rows, scores = [], []
        child = torch.zeros(B,A,7,W,device=ctx.device,dtype=ctx.dtype)
        for y in range(ROWS-1,-1,-1):
            room = self.room(b['graph_room'][:,y])[:,None].expand(-1,A,-1,-1)
            floor = self.floor.weight[y].expand(B,A,7,-1)
            c = ctx[:,:,None].expand(-1,-1,7,-1)
            num = b['graph_numeric'][:,y,None].expand(-1,A,-1,-1)
            node = self.node_mlp(torch.cat([room,floor,c,num],-1))
            if y < ROWS-1:
                child_score = self.branch_score(torch.cat([child,c],-1)).squeeze(-1)
                successors = child[:,:,None].expand(-1,-1,7,-1,-1)
                edge = b['graph_edge'][:,y,None].expand(-1,A,-1,-1)
                summary = alternatives(successors,edge,child_score[:,:,None].expand(-1,-1,7,-1))
                node = self.node_norm(node+self.node_update(torch.cat([node,summary],-1)))
            node = node*b['graph_mask'][:,y,None,:,None]
            score = self.branch_score(torch.cat([node,c],-1)).squeeze(-1)
            rows.append(node); scores.append(score)
            child = node
        nodes = torch.stack(rows[::-1],2)
        score = torch.stack(scores[::-1],2)
        flat = nodes.flatten(2,3)
        roots = b['graph_roots'][:,None].expand(-1,A,-1,-1).flatten(2,3)
        summary = alternatives(flat,roots,score.flatten(2,3))
        return summary,score

    def forward(self,b):
        if 'graph_room' not in b:
            raise ValueError('run_policy_v3 requires encode(..., kind=model.KIND) and new graph observations')
        ctx = self._context(b)
        summary,_ = self.map_features(b,ctx)
        h = self.head_in(torch.cat([ctx,summary],-1))
        for block in self.blocks:
            h = h+block(h)
        return (self.win_out(self.head_norm(h)).squeeze(-1),)


class RunPolicyV3Attention(RunPolicyV3):
    KIND = 'run_policy_v3_2'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = nn.ModuleList([nn.TransformerEncoderLayer(self.args['width'],self.args['heads'],
                                      2*self.args['width'],dropout=self.args['dropout'],
                                      batch_first=True,norm_first=True)])
