"""Offline comparison of overworld value checkpoints on held-out runs (same split as value/learn.py).
Usage: offline_eval.py --data DIR... --models CKPT... [--names a b] [--target full] [--json OUT] [--all]"""
import argparse
import json
import math
import random

import torch

from agents.overworld.value.core import encode, load_model, node_bosses, nodes, read_runs, run_score


def held_out(run):  # identical to learn.py
    return random.Random(run["seed"]).random() < 0.1


def collect(runs, target):
    ex = []  # (state, options, column, boss, g, act, is_act_start)
    for run in runs:
        g, seen = run_score(run, target), set()
        for (s, o, c), b in zip(nodes(run), node_bosses(run)):
            act = s.get("act", 1)
            ex.append((s, o, c, b, g, act, act not in seen))
            seen.add(act)
    return ex


def predict(model, ex, device):
    graph = model.KIND.startswith("run_policy_v3")
    size, out = (64 if graph else 512), []
    with torch.no_grad():
        for i in range(0, len(ex), size):
            chunk = ex[i:i + size]
            b = encode([(e[0], e[1]) for e in chunk], [e[3] for e in chunk], kind=model.KIND)
            b = {k: v.to(device) for k, v in b.items()}
            K = b["opt_id"].shape[1]
            col = torch.tensor([K if e[2] is None else e[2] for e in chunk], device=device)
            out += torch.sigmoid(model(b)[0].gather(1, col[:, None]).squeeze(1)).tolist()
    return out


def stats(p, g):
    n = len(g)
    if n == 0:
        return None
    eps = 1e-7
    bce = -sum(y * math.log(min(max(q, eps), 1 - eps)) + (1 - y) * math.log(1 - min(max(q, eps), 1 - eps)) for q, y in zip(p, g)) / n
    mse = sum((q - y) ** 2 for q, y in zip(p, g)) / n
    mu = sum(g) / n
    var = sum((y - mu) ** 2 for y in g) / n
    return {"n": n, "bce": bce, "mse": mse, "r2": 1 - mse / var if var > 0 else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--target", default="full", choices=["act1", "floors", "floors3", "full", "heart"])
    ap.add_argument("--json")
    ap.add_argument("--all", action="store_true", help="evaluate all runs (testing; not held-out)")
    a = ap.parse_args()
    names = a.names or [m for m in a.models]
    if len(names) != len(a.models):
        ap.error("--names must match --models")
    runs = read_runs(a.data)
    if not a.all:
        runs = [r for r in runs if held_out(r)]
    ex = collect(runs, a.target)
    g = [e[4] for e in ex]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(runs)} runs, {len(ex)} nodes ({'all' if a.all else 'held-out'}), target {a.target}, mean g {sum(g) / max(len(g), 1):.3f}")
    groups = {"all": lambda e: True, **{f"act{k}": (lambda e, k=k: e[5] == k) for k in (1, 2, 3, 4)},
              "start": lambda e: e[6], **{f"start{k}": (lambda e, k=k: e[6] and e[5] == k) for k in (2, 3)}}
    res = {}
    for name, path in zip(names, a.models):
        model = load_model(path, device)
        p = predict(model, ex, device)
        res[name] = {gn: stats([q for q, e in zip(p, ex) if f(e)], [e[4] for e in ex if f(e)]) for gn, f in groups.items()}
    for metric in ("bce", "mse", "r2"):
        print(f"\n{metric}   " + " ".join(f"{gn}(n={next(iter(res.values()))[gn]['n'] if next(iter(res.values()))[gn] else 0})".rjust(16) for gn in groups))
        for name in names:
            print(f"{name[:12]:12s}" + " ".join((f"{res[name][gn][metric]:.4f}" if res[name][gn] else "-").rjust(16) for gn in groups))
    if a.json:
        json.dump({"target": a.target, "runs": len(runs), "nodes": len(ex), "models": res}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
