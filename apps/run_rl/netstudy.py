#!/usr/bin/env python3
"""Offline network screen (slop_docs/run-rl/network-study.md): every variant trains on the SAME fixed runs with
Monte Carlo targets (the run's score: a fixed target, unlike TD), split by seed; report held-out BCE / Brier.

  netstudy.py --data DIR... --out OUT.jsonl --variants NAME=JSON ... [--seeds 0 1] [--epochs 20]
Offline loss only screens; real games decide (see the study doc).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_model, encode, nodes, read_runs, score  # noqa: E402


def aux_targets(r):
    """Run-level auxiliary targets in [0, 1]: HP fraction entering the boss (0 if not reached), reached boss, floor/16."""
    boss = [s for s in r["steps"] if s["kind"] == "fight" and s["category"] == "boss"]
    mx = max(r["steps"][0]["state"]["max_hp"], 1)
    return [boss[0]["hp_before"] / mx if boss else 0.0, float(bool(boss)), min(r["floor"], 16) / 16]


def examples(runs, progress):
    ex = []
    for r in runs:
        g = score(r, progress)
        aux = aux_targets(r)
        for k, (s, o, c) in enumerate(nodes(r)):
            ex.append((s, o, c, r["boss"], g, r["seed"], aux))
    return ex


SMALL = {"card_id": torch.int16, "opt_id": torch.int16, "relic_id": torch.int16, "potion_id": torch.int16,
         "path_room": torch.uint8, "boss": torch.uint8}


def encode_all(ex, size):
    out = []
    for i in range(0, len(ex), size):
        chunk = ex[i:i + size]
        b = encode([(s, o) for s, o, *_ in chunk], [e[3] for e in chunk])
        b = {k: v.to(SMALL[k]) if k in SMALL else v for k, v in b.items()}
        K = b["opt_id"].shape[1]
        col = torch.tensor([K if e[2] is None else e[2] for e in chunk])
        y = torch.tensor([e[4] for e in chunk])
        pick = torch.tensor([e[2] is not None or len(e[1]) > 0 for e in chunk])
        aux = torch.tensor([e[6] for e in chunk])
        out.append((b, col, y, pick, aux))
    return out


def to(b, device):
    return {k: v.to(device, non_blocking=True).long() if k in SMALL else v.to(device, non_blocking=True)
            for k, v in b.items()}


def evaluate(model, data, device):
    model.eval()
    s = {"bce": 0.0, "brier": 0.0, "n": 0, "pick_bce": 0.0, "pick_n": 0}
    with torch.no_grad():
        for b, col, y, pick, _ in data:
            logit = model(to(b, device))[0].gather(1, col.to(device)[:, None]).squeeze(1).float()
            y = y.to(device)
            bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
            s["bce"] += bce.sum().item(); s["brier"] += ((torch.sigmoid(logit) - y) ** 2).sum().item(); s["n"] += len(y)
            pk = pick.to(device)
            s["pick_bce"] += bce[pk].sum().item(); s["pick_n"] += pk.sum().item()
    return {"bce": s["bce"] / s["n"], "brier": s["brier"] / s["n"], "pick_bce": s["pick_bce"] / max(s["pick_n"], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--progress", type=float, default=0.25)
    ap.add_argument("--gpu-fraction", type=float, default=0.4)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    a = ap.parse_args()
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)  # leave room for the RL loop's training
    # one file at a time: the raw JSON of all runs does not fit in memory
    t0 = time.time()
    train, val, ytr, yva, nruns = [], [], [], [], 0
    for d in a.data:
        runs = read_runs([d])
        nruns += len(runs)
        held = {r["seed"] for r in runs if random.Random(r["seed"] * 7 + 1).random() < 0.15}
        ex = examples(runs, a.progress)
        del runs
        tr = [e for e in ex if e[5] not in held]
        va = [e for e in ex if e[5] in held]
        random.Random(0).shuffle(tr)
        train += encode_all(tr, 512); val += encode_all(va, 512)
        ytr += [e[4] for e in tr]; yva += [e[4] for e in va]
        del ex, tr, va
    mean = sum(ytr) / len(ytr)
    import math
    const = sum(-(y * math.log(mean) + (1 - y) * math.log(1 - mean)) for y in yva) / len(yva)
    print(f"{nruns} runs, {len(ytr)} train / {len(yva)} val nodes, encoded in {time.time() - t0:.0f}s; "
          f"constant-predictor val bce {const:.4f}", flush=True)
    for spec in a.variants:
        name, arch = spec.split("=", 1)
        for seed in a.seeds:
            torch.manual_seed(seed); random.seed(seed)
            model = build_model(json.loads(arch)).to(device)
            params = sum(p.numel() for p in model.parameters())
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
            best, best_ep, bad, t1 = None, 0, 0, time.time()
            for ep in range(1, a.epochs + 1):
                model.train()
                order = list(range(len(train))); random.shuffle(order)
                for i in order:
                    b, col, y, _, aux = train[i]
                    out = model(to(b, device))
                    col = col.to(device)
                    logit = out[0].gather(1, col[:, None]).squeeze(1)
                    loss = F.binary_cross_entropy_with_logits(logit, y.to(device))
                    if len(out) > 1:
                        al = out[1].gather(1, col[:, None, None].expand(-1, 1, out[1].shape[-1])).squeeze(1)
                        loss = loss + a.aux_weight * F.binary_cross_entropy_with_logits(al, aux.to(device))
                    opt.zero_grad(); loss.backward(); opt.step()
                m = evaluate(model, val, device)
                if best is None or m["bce"] < best["bce"] - 1e-5:
                    best, best_ep, bad = m, ep, 0
                    torch.save({"kind": model.KIND, "args": model.args, "state_dict": model.state_dict()},
                               Path(a.out).with_suffix(f".{name}.s{seed}.pt"))
                else:
                    bad += 1
                    if bad >= 3:
                        break
            row = {"variant": name, "seed": seed, "arch": json.loads(arch), "params": params, "best_epoch": best_ep,
                   **best, "const_bce": const, "minutes": (time.time() - t1) / 60}
            print(json.dumps(row), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
