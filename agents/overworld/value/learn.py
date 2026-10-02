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


from agents.overworld.value.core import all_node_values, build_model, encode, load_model, node_bosses, nodes, read_runs, save_model, run_score, td_targets, LAST_FLOOR, LAST_FLOOR3  # noqa: E402


def aux_targets(r, last_floor=16):
    """As netstudy.py: HP fraction entering the (first) boss (0 if not reached), reached it, floor / last_floor."""
    boss = [s for s in r["steps"] if s["kind"] == "fight" and s["category"] == "boss"]
    mx = max(r["steps"][0]["state"]["max_hp"], 1)
    return [boss[0]["hp_before"] / mx if boss else 0.0, float(bool(boss)), min(r["floor"], last_floor) / last_floor]


def build_examples(runs, weights, init, lam, progress, hp, device, target="act1"):
    values = all_node_values(init, runs, device) if init is not None and lam < 1 else None
    ex = []
    for k, run in enumerate(runs):
        ns = nodes(run)
        if not ns:
            continue
        g = run_score(run, target, progress, hp)
        targets = [g] * len(ns) if values is None else td_targets(values[k], g, lam)
        aux = aux_targets(run, 16 if target == "act1" else LAST_FLOOR3 if target == "floors3" else LAST_FLOOR)
        ex += [(s, o, c, b, t, run["seed"], weights[k], aux) for (s, o, c), t, b in zip(ns, targets, node_bosses(run))]
    return ex


SMALL = {"card_id": torch.int16, "opt_id": torch.int16, "relic_id": torch.int16, "potion_id": torch.int16,
         "path_room": torch.uint8, "boss": torch.uint8}


def to(b, device):
    return {k: v.to(device, non_blocking=True).long() if k in SMALL else v.to(device, non_blocking=True)
            for k, v in b.items()}


def batches(ex, size, device, shuffle):
    """Encoded minibatches (b, column, target, weight), kept on the CPU in compact dtypes (moved per step: all of them
    on the GPU overflow its memory); encode once, reuse every epoch."""
    idx = list(range(len(ex)))
    if shuffle:
        random.shuffle(idx)
    out = []
    for i in range(0, len(idx), size):
        chunk = [ex[j] for j in idx[i:i + size]]
        b = encode([(s, o) for s, o, *_ in chunk], [e[3] for e in chunk])
        b = {k: v.to(SMALL[k]) if k in SMALL else v for k, v in b.items()}
        K = b["opt_id"].shape[1]
        col = torch.tensor([K if e[2] is None else e[2] for e in chunk])
        y = torch.tensor([e[4] for e in chunk])
        w = torch.tensor([e[6] for e in chunk])
        aux = torch.tensor([e[7] for e in chunk])
        out.append((b, col, y, w, aux))
    return out


def evaluate(model, val, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for b, col, y, _, _ in val:
            col, y = col.to(device), y.to(device)
            logit = model(to(b, device))[0].gather(1, col[:, None]).squeeze(1)
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
    ap.add_argument("--target", choices=["act1", "floors", "floors3"], default="act1",
                    help="act1: clear score (--progress/--hp); floors: two-act floor score; floors3: three-act floor score")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--td-model", help="model giving the TD bootstrap values (default: --init)")
    ap.add_argument("--aux-weight", type=float, default=0.5, help="weight of auxiliary targets (models with aux > 0)")
    ap.add_argument("--arch", default="{}", help='new model (no --init): {"kind": ..., args}; default run_policy_v1')
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
    td = load_model(a.td_model, device) if a.td_model else init
    ex = build_examples(runs, weights, td, a.lam, a.progress, a.hp, device, a.target)
    held = {r["seed"] for r in runs if random.Random(r["seed"]).random() < 0.1}
    train = [e for e in ex if e[5] not in held]
    val = [e for e in ex if e[5] in held]
    print(f"{len(runs)} runs, {len(train)} train nodes, {len(val)} val nodes; "
          f"mean score {sum(run_score(r, a.target, a.progress, a.hp) for r in runs) / len(runs):.3f}", flush=True)
    model = (build_model({"kind": init.KIND, **init.args}) if init else build_model(json.loads(a.arch))).to(device)
    if init:
        model.load_state_dict(init.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    best, best_state, bad = float("inf"), None, 0
    val = batches(val, 512, device, False)
    train = batches(train, 256, device, True)
    base = evaluate(model, val, device)
    print(f"epoch 0 val {base:.4f}", flush=True)
    for epoch in range(1, a.epochs + 1):
        model.train()
        random.shuffle(train)
        for b, col, y, w, aux in train:
            col, y, w = col.to(device), y.to(device), w.to(device)
            out = model(to(b, device))
            logit = out[0].gather(1, col[:, None]).squeeze(1)
            loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / w.sum()
            if getattr(model, "aux_out", None) is not None:  # run_policy_v2 aux head only (v1's extra heads are untrained)
                al = out[1].gather(1, col[:, None, None].expand(-1, 1, out[1].shape[-1])).squeeze(1)
                lx = F.binary_cross_entropy_with_logits(al, aux.to(device), reduction="none").mean(1)
                loss = loss + a.aux_weight * (lx * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        v = evaluate(model, val, device)
        print(f"epoch {epoch} val {v:.4f}", flush=True)
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(best_state)
    save_model(model.cpu(), a.out, data=a.data, target=a.target, lam=a.lam, progress=a.progress, hp=a.hp, decay=a.decay, val_bce=best,
               runs=len(runs))
    print(json.dumps({"out": a.out, "val_bce": best, "runs": len(runs)}))


if __name__ == "__main__":
    main()
