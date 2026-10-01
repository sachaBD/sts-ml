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

from models.combat_outcome.learn import batch
from models.combat_outcome.learn_marginals import KINDS

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'build/main/card_search_worker'
CHECKPOINTS = [ROOT / f'runs/schema=combat_outcome_v1/date=2026-09-30/id=tpair1-co2-w32-h64-l1-d30-lr.001-s{s}/out/augmented.pt'
               for s in range(3)]
from agents.overworld.search.policy import Ensemble, answer_fights, answer_picks


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
