#!/usr/bin/env python3
"""The real-run RL loop (docs/research/run-rl/README.md). Resumable: a step whose output exists is skipped.

  loop.py --root runs/run_rl/<name> [--iters 20] [--batch 2000] [--eval-seeds 500] [--workers 11]
          [--eps 0.1] [--eps0 0.2] [--lam 0.7] [--progress 0.25] [--hp 0.0] [--window 3]

  baseline/          SimpleAgent picks on the eval seeds (once)
  iter000/data       SimpleAgent picks + eps0 random exploration on fresh training seeds
  iter000/model.pt   V trained on iter000 (Monte Carlo targets)
  iterNNN/data       net picks (iter N-1 model) + eps exploration
  iterNNN/model.pt   V trained on the last `window` batches, TD(lam) targets from the iter N-1 model
  iterNNN/eval       greedy net picks (iterNNN model) on the eval seeds -> clear rate vs baseline (curve.tsv)
Eval seeds: 600000000000.. (the act1-eval-mcts seeds); training seeds: 700000000000 + iter * 100000 ..
"""
import argparse
import json
import shutil
import subprocess
import time
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
EVAL_SEED = 600_000_000_000
TRAIN_SEED = 700_000_000_000


def run(cmd):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


WORKER = None  # the run's own copy of build/main/run_rl_worker (set in main)


DECIDE = []  # --decide (set in main)


def play(out, first, n, workers, policy, ckpt=None, eps=0.0):
    if (out / "summary.json").exists():
        return
    # an incomplete runs.jsonl is resumed by play.py
    cmd = [PY, HERE / "play.py", "--out", out, "--first-seed", first, "--seeds", n, "--workers", workers,
           "--policy", policy, "--eps", eps, "--worker", WORKER]
    if DECIDE:
        cmd += ["--decide", *DECIDE]
    run(cmd + (["--ckpt", ckpt] if ckpt else []))


def paired(eval_dir, base_dir):
    load = lambda d: {r["seed"]: r["status"] == "act_complete" for r in map(json.loads, open(d / "runs.jsonl"))}
    a, b = load(eval_dir), load(base_dir)
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    mean = sum(d) / n
    se = (sum((x - mean) ** 2 for x in d) / (n - 1) / n) ** 0.5
    return sum(a[s] for s in seeds) / n, sum(b[s] for s in seeds) / n, mean, se, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--eval-seeds", type=int, default=500)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--eps0", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--progress", type=float, default=0.25)
    ap.add_argument("--hp", type=float, default=0.0)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--decay", type=float, default=1.0)
    ap.add_argument("--worker", default="build/main/run_rl_worker", help="copied into ROOT once (first start)")
    ap.add_argument("--decide", nargs="*", default=[], choices=["rest", "path", "shop", "neow", "event"], help="policy-made decisions besides cards")
    ap.add_argument("--init", help="start from this model (iter 0 plays it with eps) instead of SimpleAgent + eps0")
    ap.add_argument("--extra-data", nargs="*", default=[], help="older data dirs placed before the loop's own in the window")
    ap.add_argument("--arch", default="{}", help="architecture of a fresh model (no --init)")
    ap.add_argument("--seed-offset", type=int, default=0, help="added to the training seeds (fresh seeds per loop)")
    a = ap.parse_args()
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "config.json", "a") as f:  # appended on every (re)start
        f.write(json.dumps({**vars(a), "started": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    global WORKER, DECIDE
    DECIDE = a.decide
    WORKER = root / "run_rl_worker"
    if not WORKER.exists():
        shutil.copy2(a.worker, WORKER)
    reward = ["--lam", a.lam, "--progress", a.progress, "--hp", a.hp]

    base = root / "baseline"
    play(base, EVAL_SEED, a.eval_seeds, a.workers, "simple")
    prev = None
    for i in range(a.iters):
        it = root / f"iter{i:03d}"
        data = it / "data"
        if i == 0 and a.init:
            play(data, TRAIN_SEED + a.seed_offset, a.batch, a.workers, "net", a.init, a.eps)
        elif i == 0:
            play(data, TRAIN_SEED + a.seed_offset, a.batch, a.workers, "simple", eps=a.eps0)
        else:
            play(data, TRAIN_SEED + a.seed_offset + i * 100_000, a.batch, a.workers, "net", prev, a.eps)
        model = it / "model.pt"
        if not model.exists():
            dirs = [Path(d) for d in a.extra_data] + [root / f"iter{j:03d}" / "data" for j in range(i + 1)]
            window = dirs[-a.window:]
            init = prev or a.init
            run([PY, HERE / "train.py", "--data", *window, "--out", model, *reward, "--decay", a.decay]
                + (["--init", init] if init else ["--arch", a.arch]))
        play(it / "eval", EVAL_SEED, a.eval_seeds, a.workers, "net", model)
        net, simple, diff, se, n = paired(it / "eval", base)
        line = f"{i}\t{net:.3f}\t{simple:.3f}\t{diff:+.3f}\t{se:.3f}\t{n}"
        curve = root / "curve.tsv"
        if not curve.exists():
            curve.write_text("iter\tnet_clear\tsimple_clear\tdiff\tse_diff\tseeds\n")
        if line.split("\t")[0] not in [l.split("\t")[0] for l in curve.read_text().splitlines()[1:]]:
            with open(curve, "a") as f:
                f.write(line + "\n")
        print("CURVE", line, flush=True)
        prev = model


if __name__ == "__main__":
    main()
