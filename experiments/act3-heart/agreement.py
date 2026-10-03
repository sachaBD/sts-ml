#!/usr/bin/env python3
"""Decision agreement of overworld models with the choices logged in a play dir (e.g. the incumbent's greedy dev
play): per decision type x act, share of decisions where each model's argmax equals the logged choice, plus card
skip rates. Usage: agreement.py --data DIR --models CKPT... --names ... [--max-runs N]"""
import argparse, collections, torch
from agents.overworld.value.core import encode, load_model, iter_runs

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--names", nargs="+"); ap.add_argument("--max-runs", type=int, default=300)
a = ap.parse_args()
torch.set_num_threads(2)
models = [load_model(m) for m in a.models]; names = a.names or a.models
agree = collections.defaultdict(lambda: [0] * (len(models) + 1))  # key -> [n, agree per model]
skips = collections.defaultdict(lambda: [0] * (len(models) + 2))  # [n, logged skips, per model]
for k, (_, run) in enumerate(iter_runs([a.data])):
    if k >= a.max_runs: break
    for s in run["steps"]:
        if s["kind"] == "pick" and s.get("source") == "net":
            n = len(s["options"]); key = ("card", s["state"]["act"])
            agree[key][0] += 1; skips[key][0] += 1; skips[key][1] += s["choice"] >= n
            for i, m in enumerate(models):
                with torch.no_grad():
                    v = m(encode([(s["state"], s["options"])], [s["boss"]], kind=m.KIND))[0][0]
                c = int(torch.argmax(v[: n + 1]).item()) if v.shape[0] > n else int(torch.argmax(v).item())
                c = n if c >= n else c
                agree[key][i + 1] += c == s["choice"]; skips[key][i + 2] += c == n
        elif s["kind"] == "decide" and s.get("source") == "net" and isinstance(s.get("after"), list) and s["after"]:
            key = (s["decision"], (s["after"][0] or {}).get("act", 0))
            agree[key][0] += 1
            for i, m in enumerate(models):
                with torch.no_grad():
                    v = m(encode([(x, []) for x in s["after"]], [s["boss"]] * len(s["after"]), kind=m.KIND))[0][:, -1]
                agree[key][i + 1] += int(torch.argmax(v).item()) == s["choice"]
print("agreement with logged choice (share); key = (decision, act)")
print(f"{'key':22s} {'n':>5s} " + " ".join(f"{x:>12s}" for x in names))
for key in sorted(agree, key=str):
    n, *c = agree[key]
    print(f"{str(key):22s} {n:5d} " + " ".join(f"{x / n:12.3f}" for x in c))
print("card skip rate: logged, then models")
for key in sorted(skips, key=str):
    n, *c = skips[key]
    print(f"{str(key):22s} {n:5d} " + " ".join(f"{x / n:12.3f}" for x in c))
