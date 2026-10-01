#!/usr/bin/env python3
"""Play real act 1 runs (MCTS fights, SimpleAgent out of combat) with card picks from a policy; log every run.

  play.py --out DIR --first-seed S --seeds N [--workers W] --policy simple|random|net [--ckpt PATH]
          [--eps E] [--explore random|simple] [--sims easy,hard,elite,event,boss]

policy simple: SimpleAgent's pick.  random: uniform over cards + skip.  net: argmax after-state value of --ckpt.
eps: with probability eps pick uniformly at random instead (exploration). The choice's source is logged.
Writes DIR/runs.jsonl (one run per line; map nodes dropped) and DIR/summary.json.
"""
import argparse
import json
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.overworld.value.core import encode, load_model  # noqa: E402

WORKER = Path("build/main/run_rl_worker")


from agents.overworld.value.policy import Policy


def strip(state):
    state = dict(state)
    state["map"] = {k: v for k, v in state["map"].items() if k != "nodes"}
    return state


def worker_loop(worker, seeds, lock, policy, sims, decide, lookahead, out, stats):
    proc = subprocess.Popen([str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        while True:
            with lock:
                seed = next(seeds, None)
            if seed is None:
                return
            proc.stdin.write(json.dumps({"seed": seed, "ascension": 20, "simulations": sims, "decide": decide,
                                         "lookahead_samples": lookahead[0], "lookahead_horizon": lookahead[1]}) + "\n")
            proc.stdin.flush()
            meta = []
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"worker exited during seed {seed}")
                msg = json.loads(line)
                if msg["type"] == "pick":
                    choice, source, vals = policy(msg)
                    if not msg.get("lookahead"):
                        meta.append({"source": source, "values": vals})
                    proc.stdin.write(json.dumps({"choice": choice}) + "\n")
                    proc.stdin.flush()
                elif msg["type"] in ("decide", "evaluate"):
                    choice, source, vals = (policy.evaluate if msg["type"] == "evaluate" else policy.decide)(msg)
                    if choice >= 0 and not msg.get("lookahead"):
                        meta.append({"source": source, "values": vals})
                    proc.stdin.write(json.dumps({"choice": choice}) + "\n")
                    proc.stdin.flush()
                elif msg["type"] == "done":
                    break
                else:
                    raise RuntimeError(f"worker error on seed {seed}: {msg}")
            picks = [s for s in msg["steps"] if s["kind"] in ("pick", "decide")]
            for s, m in zip(picks, meta):
                s.update(m)
            for s in msg["steps"]:
                for key in ("state", "after"):
                    if key in s:
                        s[key] = strip(s[key])
            with lock:
                out.write(json.dumps(msg) + "\n")
                out.flush()
                stats["runs"] += 1
                stats["clear"] += msg["status"] == "act_complete"
                stats["seconds"] += msg["seconds"]
    finally:
        proc.stdin.close()
        proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-seed", type=int, required=True)
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--policy", choices=["simple", "random", "net"], required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--eps", type=float, default=0.0)
    ap.add_argument("--sims", default="500,2000,5000,5000,15000")
    ap.add_argument("--decide", nargs="*", default=[], choices=["rest", "path", "shop", "neow", "event"],
                    help="decisions besides card picks made by the policy (else SimpleAgent)")
    ap.add_argument("--samples", type=int, default=8, help="lookahead samples per event / Neow option")
    ap.add_argument("--horizon", type=int, default=0, help="lookahead floors (0 = until the event resolves)")
    ap.add_argument("--worker", default=str(WORKER), help="run_rl_worker binary (loop.py passes its own snapshot)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    sims = dict(zip(["easy", "hard", "elite", "event", "boss"], map(int, a.sims.split(","))))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = Policy(a.policy, a.ckpt, a.eps, seed=a.first_seed)
    # Resume: keep complete lines of an earlier, interrupted invocation and skip their seeds.
    stats = {"runs": 0, "clear": 0, "seconds": 0.0}
    done, kept = set(), []
    path = out_dir / "runs.jsonl"
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(r["seed"]); kept.append(line if line.endswith("\n") else line + "\n")
            stats["runs"] += 1; stats["clear"] += r["status"] == "act_complete"; stats["seconds"] += r["seconds"]
        path.write_text("".join(kept))
        print(f"resuming: {len(done)} runs already played", flush=True)
    seeds = iter([s for s in range(a.first_seed, a.first_seed + a.seeds) if s not in done])
    lock = threading.Lock()
    t0 = time.monotonic()
    with open(out_dir / "runs.jsonl", "a") as out:
        threads = [threading.Thread(target=worker_loop, args=(a.worker, seeds, lock, policy, sims, a.decide, (a.samples, a.horizon), out, stats))
                   for _ in range(a.workers)]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            time.sleep(30)
            with lock:
                r = max(stats["runs"], 1)
                print(f"{time.monotonic() - t0:7.0f}s  runs {stats['runs']}/{a.seeds}  clear {stats['clear'] / r:.3f}"
                      f"  {stats['seconds'] / r:.1f} worker-s/run", flush=True)
        for t in threads:
            t.join()
    summary = {**vars(a), "sims": sims, **stats, "wall_seconds": time.monotonic() - t0,
               "clear_rate": stats["clear"] / max(stats["runs"], 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
