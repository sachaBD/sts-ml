#!/usr/bin/env python3
"""Train the run value V (run_policy_v1 value head) on logged real runs with TD(lambda) targets.

  train.py --data DIR [DIR ...] --out CKPT [--init CKPT] [--lam 0.7] [--progress 0.25] [--hp 0.0] [--epochs 30]

Targets: each run's nodes (common.nodes) get TD(lambda) targets from the run's score (common.score) and the
values of the NEXT nodes under the --init model; --decay weights older --data dirs less (lam = 1 or no --init: pure Monte Carlo, target = score).
Loss: BCE(sigmoid(win_logit[node column]), target). Split by seed (10% held out) for early stopping.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import all_node_values, encode, load_model, nodes, read_runs, save_model, score, td_targets  # noqa: E402
from sts_combat_rl.topology.run_policy_v1 import RunPolicyV1  # noqa: E402


def build_examples(runs, weights, init, lam, progress, hp, device):
    values = all_node_values(init, runs, device) if init is not None and lam < 1 else None
    ex = []
    for k, run in enumerate(runs):
        ns = nodes(run)
        if not ns:
            continue
        g = score(run, progress, hp)
        targets = [g] * len(ns) if values is None else td_targets(values[k], g, lam)
        ex += [(s, o, c, run["boss"], t, run["seed"], weights[k]) for (s, o, c), t in zip(ns, targets)]
    return ex


def batches(ex, size, device, shuffle):
    """Encoded minibatches (b, column, target, weight); encode once, reuse every epoch."""
    idx = list(range(len(ex)))
    if shuffle:
        random.shuffle(idx)
    out = []
    for i in range(0, len(idx), size):
        chunk = [ex[j] for j in idx[i:i + size]]
        b = encode([(s, o) for s, o, *_ in chunk], [e[3] for e in chunk], device)
        K = b["opt_id"].shape[1]
        col = torch.tensor([K if e[2] is None else e[2] for e in chunk], device=device)
        y = torch.tensor([e[4] for e in chunk], device=device)
        w = torch.tensor([e[6] for e in chunk], device=device)
        out.append((b, col, y, w))
    return out


def evaluate(model, val):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b, col, y, _ in val:
            logit = model(b)[0].gather(1, col[:, None]).squeeze(1)
            tot += F.binary_cross_entropy_with_logits(logit, y, reduction="sum").item()
            n += len(y)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--progress", type=float, default=0.25)
    ap.add_argument("--hp", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decay", type=float, default=1.0,
                    help="loss weight of a --data dir = decay ** (dirs after it); 1 = all equal")
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs, weights = [], []
    for k, d in enumerate(a.data):
        rs = read_runs([d])
        runs += rs
        weights += [a.decay ** (len(a.data) - 1 - k)] * len(rs)
    init = load_model(a.init, device) if a.init else None
    ex = build_examples(runs, weights, init, a.lam, a.progress, a.hp, device)
    held = {r["seed"] for r in runs if random.Random(r["seed"]).random() < 0.1}
    train = [e for e in ex if e[5] not in held]
    val = [e for e in ex if e[5] in held]
    print(f"{len(runs)} runs, {len(train)} train nodes, {len(val)} val nodes; "
          f"mean score {sum(score(r, a.progress, a.hp) for r in runs) / len(runs):.3f}", flush=True)
    model = RunPolicyV1(**(init.args if init else {})).to(device)
    if init:
        model.load_state_dict(init.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    best, best_state, bad = float("inf"), None, 0
    val = batches(val, 1024, device, False)
    train = batches(train, 256, device, True)
    base = evaluate(model, val)
    print(f"epoch 0 val {base:.4f}", flush=True)
    for epoch in range(1, a.epochs + 1):
        model.train()
        random.shuffle(train)
        for b, col, y, w in train:
            logit = model(b)[0].gather(1, col[:, None]).squeeze(1)
            loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        v = evaluate(model, val)
        print(f"epoch {epoch} val {v:.4f}", flush=True)
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(best_state)
    save_model(model.cpu(), a.out, data=a.data, lam=a.lam, progress=a.progress, hp=a.hp, decay=a.decay, val_bce=best,
               runs=len(runs))
    print(json.dumps({"out": a.out, "val_bce": best, "runs": len(runs)}))


if __name__ == "__main__":
    main()
