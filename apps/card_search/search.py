"""Card-pick search driver: runs card_search_worker processes and answers their batched model requests on the GPU.

Fights: sampled from the combat outcome model ensemble (member chosen by the worker; common random numbers u1/u2).
Picks inside rollouts and on the way to the root: policy v0 = argmax over options of the outcome model's expected
score (P(win) * E[HP | win]) averaged over the act boss (weight 2) and the three act 1 elites, at current HP.

usage: search.py OUT.jsonl --seeds 1 2 ... --decisions 0 2 --rollouts 200 --reps 2 [--workers 4]
Each (seed, decision, rep) is one search: same root (root_seed = seed), independent rollouts (rollout_seed = rep).
"""
import argparse
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from apps.combat_transition.train import batch
from apps.combat_transition.train_marginals import KINDS

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'build/main/card_search_worker'
CHECKPOINTS = [ROOT / f'runs/schema=combat_outcome_v1/date=2026-09-30/id=tpair1-co2-w32-h64-l1-d30-lr.001-s{s}/out/augmented.pt'
               for s in range(3)]
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


def run_jobs(model, jobs, results, lock, out):
    proc = subprocess.Popen([str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        for job in jobs:
            proc.stdin.write(json.dumps(job) + '\n'); proc.stdin.flush()
            while True:
                msg = json.loads(proc.stdout.readline())
                if msg['type'] == 'result':
                    msg['job'] = job
                    with lock:
                        out.write(json.dumps(msg) + '\n'); out.flush(); results.append(msg)
                    if len(results) % 50 == 0: print(f"{len(results)} jobs done", flush=True)
                    break
                reply = {'fights': answer_fights(model, msg['fights']), 'picks': answer_picks(model, msg['picks'])}
                proc.stdin.write(json.dumps(reply) + '\n'); proc.stdin.flush()
    finally:
        proc.stdin.close(); proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out', type=Path); ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--decisions', type=int, nargs='+', default=[0]); ap.add_argument('--rollouts', type=int, default=200)
    ap.add_argument('--reps', type=int, default=2); ap.add_argument('--simple-picks', action='store_true'); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    model = Ensemble(CHECKPOINTS)
    jobs = [dict(seed=s, decision=d, rollouts=a.rollouts, root_seed=s, rollout_seed=r + 1, simple_picks=a.simple_picks)
            for s in a.seeds for d in a.decisions for r in range(a.reps)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    results, lock = [], threading.Lock()
    with a.out.open('x') as out, ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(lambda i: run_jobs(model, jobs[i::a.workers], results, lock, out), range(a.workers)))


if __name__ == '__main__':
    main()
