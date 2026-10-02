#!/usr/bin/env python3
"""Play real runs (Act 1, or through --max-act) with learned overworld choices and category-specific combat.

Optional --combat-out records replay-verified combat_v4 facts; --overworld-record writes overworld_v1.
--combat-leaf value_net uses neural leaves except easy fights, which stay guided-rollout.
--combat-explore applies one seeded random action per recorded fight (never inside hypothetical lookahead).

  play.py --out DIR --first-seed S --seeds N [--workers W] --policy simple|random|net [--ckpt PATH]
          [--eps E] [--explore random|simple] [--sims easy,hard,elite,event,boss]

policy simple: SimpleAgent's pick.  random: uniform over cards + skip.  net: argmax after-state value of --ckpt.
eps: with probability eps pick uniformly at random instead (exploration). The choice's source is logged.
Writes DIR/runs.jsonl (one run per line; versioned graph observations retained) and DIR/summary.json.
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
from apps.run_rl.records import write_combat, write_overworld


def strip(state):
    state = dict(state)
    # v3 needs graph connectivity in both training records and inference. Legacy records
    # may still use the compact path-only representation.
    if state.get("overworld", {}).get("version") != 1:
        state["map"] = {k: v for k, v in state["map"].items() if k != "nodes"}
    return state


def worker_loop(worker, seeds, lock, policy, sims, decide, lookahead, out, stats, combat):
    proc = subprocess.Popen([str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        while True:
            with lock:
                seed = next(seeds, None)
            if seed is None:
                return
            proc.stdin.write(json.dumps({"seed": seed, "ascension": 20, "simulations": sims, "decide": decide,
                                         "lookahead_samples": lookahead[0], "lookahead_horizon": lookahead[1],
                                         "max_act": lookahead[2], **lookahead[3], **combat[0]}) + "\n")
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
                    if msg.get("decision") == "rest_lookahead" and not msg.get("lookahead"):
                        # refines the pending rest decide just asked: one logged step, so replace its annotation
                        meta[-1] = {"source": "rest_lookahead", "values": vals, "greedy": meta[-1]}
                    elif choice >= 0 and not msg.get("lookahead"):
                        meta.append({"source": source, "values": vals})
                    # values: V per option when the net chose greedily (rest lookahead refines only those), else null
                    proc.stdin.write(json.dumps({"choice": choice, "values": vals if source == "net" else None}) + "\n")
                    proc.stdin.flush()
                elif msg["type"] == "done":
                    break
                else:
                    raise RuntimeError(f"worker error on seed {seed}: {msg}")
            records = msg.pop("combat_records", [])
            if combat[1] is not None:
                write_combat(combat[1], records, seed)
            picks = [s for s in msg["steps"] if s["kind"] in ("pick", "decide")]
            for s, m in zip(picks, meta):
                s.update(m)
            for s in msg["steps"]:
                for key in ("state", "after"):
                    if key in s:
                        s[key] = strip(s[key])
            if combat[2] is not None:
                write_overworld(combat[2], msg, seed)
            with lock:
                out.write(json.dumps(msg) + "\n")
                out.flush()
                stats["runs"] += 1
                stats["clear"] += msg["status"] == "act_complete"
                stats["seconds"] += msg["seconds"]
                acts = msg.get("acts_cleared", int(msg["status"] == "act_complete"))
                stats["acts_cleared"][acts] = stats["acts_cleared"].get(acts, 0) + 1
                stats["floor_sum"] += msg["floor"]
    except BaseException as e:
        with lock:
            stats.setdefault("errors", []).append(repr(e))
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
    ap.add_argument("--max-act", type=int, choices=[1, 2, 3, 4], default=1, help="4 = all acts plus Heart, with real key requirements")
    ap.add_argument("--rest-lookahead", help="K,S: decide rest vs the greedy choice by K sampled plays to the next fight "
                    "(fight sims x S); off by default")
    ap.add_argument("--route-p", type=float, default=0.0, help="fraction of runs (by seed) taking uniform random paths")
    ap.add_argument("--target", choices=["act1", "floors", "floors3", "heart"], default="act1", help="heart = actual Heart defeat, not Act 3 clear")
    ap.add_argument("--decide", nargs="*", default=[], choices=["rest", "path", "shop", "neow", "event", "boss_relic"],
                    help="decisions besides card picks made by the policy (else SimpleAgent)")
    ap.add_argument("--samples", type=int, default=8, help="lookahead samples per event / Neow option")
    ap.add_argument("--horizon", type=int, default=0, help="lookahead floors (0 = until the event resolves)")
    ap.add_argument("--worker", default=str(WORKER), help="run_rl_worker binary (loop.py passes its own snapshot)")
    ap.add_argument("--combat-leaf", choices=["guided_rollout", "value_net"], default="guided_rollout")
    ap.add_argument("--combat-weights")
    ap.add_argument("--combat-out", help="combat_v4 out directory; record real fights, never lookahead samples")
    ap.add_argument("--combat-explore", action="store_true", help="one random legal action per recorded fight")
    ap.add_argument("--overworld-record", action="store_true", help="write overworld_v1 canonical tables in --out")
    ap.add_argument("--collection-id", default="legacy")
    a = ap.parse_args()
    if a.combat_explore and not a.combat_out:
        ap.error("combat exploration requires --combat-out")
    if (a.combat_leaf == "value_net") != bool(a.combat_weights):
        ap.error("value_net requires --combat-weights; rollout must not receive weights")
    combat_dir = Path(a.combat_out) if a.combat_out else None
    if combat_dir is not None:
        combat_dir.mkdir(parents=True, exist_ok=True)
    combat_job = {"combat_leaf": a.combat_leaf, "record_combat": combat_dir is not None,
                  "combat_explore": a.combat_explore, "collection_id": a.collection_id}
    if a.combat_weights:
        combat_job["combat_weights"] = a.combat_weights
    torch.set_num_threads(1)
    extra = {}
    if a.rest_lookahead:
        k, scale = a.rest_lookahead.split(",")
        extra["rest_lookahead"] = {"samples": int(k), "sims_scale": float(scale)}
    sims = dict(zip(["easy", "hard", "elite", "event", "boss"], map(int, a.sims.split(","))))
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy = Policy(a.policy, a.ckpt, a.eps, seed=a.first_seed, route_p=a.route_p, target=a.target)
    # Resume: keep complete lines of an earlier, interrupted invocation and skip their seeds.
    stats = {"runs": 0, "clear": 0, "seconds": 0.0, "acts_cleared": {}, "floor_sum": 0}
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
            acts = r.get("acts_cleared", int(r["status"] == "act_complete"))
            stats["acts_cleared"][acts] = stats["acts_cleared"].get(acts, 0) + 1; stats["floor_sum"] += r["floor"]
        path.write_text("".join(kept))
        print(f"resuming: {len(done)} runs already played", flush=True)
    seeds = iter([s for s in range(a.first_seed, a.first_seed + a.seeds) if s not in done])
    lock = threading.Lock()
    t0 = time.monotonic()
    with open(out_dir / "runs.jsonl", "a") as out:
        threads = [threading.Thread(target=worker_loop, args=(a.worker, seeds, lock, policy, sims, a.decide, (a.samples, a.horizon, a.max_act, extra), out, stats, (combat_job, combat_dir, out_dir if a.overworld_record else None)))
                   for _ in range(a.workers)]
        for t in threads:
            t.start()
        while any(t.is_alive() for t in threads):
            time.sleep(30)
            with lock:
                r = max(stats["runs"], 1)
                print(f"{time.monotonic() - t0:7.0f}s  runs {stats['runs']}/{a.seeds}  clear {stats['clear'] / r:.3f}"
                      f"  acts {dict(sorted(stats['acts_cleared'].items()))}  floor {stats['floor_sum'] / r:.1f}"
                      f"  {stats['seconds'] / r:.1f} worker-s/run", flush=True)
        for t in threads:
            t.join()
    if stats.get("errors") or stats["runs"] != a.seeds:
        raise RuntimeError(f"incomplete play: {stats}")
    summary = {**vars(a), "sims": sims, **stats, "wall_seconds": time.monotonic() - t0,
               "clear_rate": stats["clear"] / max(stats["runs"], 1)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
