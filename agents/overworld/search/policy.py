"""Outcome-model ensemble and rollout choices for card-reward search."""
import threading
import torch
from models.combat_outcome.learn import batch
from models.combat_outcome.learn_marginals import KINDS

V0_FIGHTS = [('boss', 2.0), ('gremlin_nob', 1.0), ('lagavulin', 1.0), ('three_sentries', 1.0)]


class Ensemble:
    def __init__(self, paths, device='cuda'):
        self.nets, self.device, self.lock = [], device, threading.Lock()
        for p in paths:
            c = torch.load(p, map_location='cpu')
            net = KINDS[c['kind']](**c['args']); net.load_state_dict(c['state_dict']); net.to(device).eval()
            self.nets.append(net); self.enc = c['encounters']; self.centers = torch.tensor(c['centers'])

    @torch.no_grad()
    def predict(self, rows, member):
        """rows: [{'encounter', 'pre'}] -> P(win) [N], P(HP bin | win) [N, BINS] from ensemble member(s)."""
        if not rows:
            return torch.zeros(0), torch.zeros(0, len(self.centers))
        with self.lock:
            b = {k: v.to(self.device) for k, v in batch(rows, self.enc, 'cpu')[0].items()}
            members = [member] if member is not None else range(len(self.nets))
            ps, hs = zip(*[(w.sigmoid().cpu(), h.softmax(-1).cpu()) for w, h in (self.nets[m](b) for m in members)])
            return torch.stack(ps).mean(0), torch.stack(hs).mean(0)


def answer_fights(model, fights):
    out = []
    for m in {f['member'] for f in fights}:
        sel = [f for f in fights if f['member'] == m]
        p, h = model.predict([{'encounter': f['encounter'], 'pre': f['state'], 'won': 0, 'final_hp': 0} for f in sel], m)
        cdf = h.cumsum(-1)
        for i, f in enumerate(sel):
            won = f['u'][0] < p[i].item()
            b = int((cdf[i] < f['u'][1]).sum().clamp(max=len(model.centers) - 1))
            hp = max(1, min(f['state']['max_hp'], round(model.centers[b].item())))
            out.append({'k': f['k'], 'won': bool(won), 'hp': hp if won else 0})
    return out


def answer_picks(model, picks):
    rows, index = [], []
    for j, q in enumerate(picks):
        for o in range(len(q['options']) + 1):
            pre = dict(q['state'])
            if o < len(q['options']):
                pre['deck'] = q['state']['deck'] + [q['options'][o]]
            for enc, w in V0_FIGHTS:
                rows.append({'encounter': q['boss'] if enc == 'boss' else enc, 'pre': pre, 'won': 0, 'final_hp': 0}); index.append((j, o, w))
    p, h = model.predict(rows, None)
    score = p * (h * model.centers).sum(-1)
    best = {}
    tot = {}
    for (j, o, w), s in zip(index, score.tolist()):
        tot[(j, o)] = tot.get((j, o), 0.0) + w * s
    for (j, o), s in tot.items():
        if j not in best or s > best[j][1]:
            best[j] = (o, s)
    return [{'k': q['k'], 'choice': best[j][0]} for j, q in enumerate(picks)]
