"""Shared pieces of the real-run RL loop (docs/research/run-rl/README.md): state encoding for run_policy_v1, the
reward / score of a finished run, TD(lambda) targets, checkpoints.

A logged run (play.py, one JSON line) = {seed, boss, status, floor, final_hp, steps: [start | fight | pick ...]}.
Training NODES are the non-terminal steps. Each node has one after-state value:
  start / fight: V(state)                    = the skip column of run_policy_v1 (no offered cards)
  pick:          V(state + chosen option)    = column `choice` (skip = the last column)
  decide:        V(after-state of the chosen rest / path option) = skip column of that state
"""
import json
from pathlib import Path

import numpy as np
import torch

from agents.overworld.value.run_policy_v1 import NONE, RunPolicyV1
from agents.overworld.value.run_policy_v2 import RunPolicyV2

KINDS = {RunPolicyV1.KIND: RunPolicyV1, RunPolicyV2.KIND: RunPolicyV2}


def build_model(arch):
    """arch: {"kind": ..., constructor args...}; kind defaults to run_policy_v1."""
    arch = dict(arch)
    return KINDS[arch.pop("kind", RunPolicyV1.KIND)](**arch)

BOSSES = {"slime_boss": 0, "the_guardian": 1, "hexaghost": 2,  # the CURRENT act's boss (gc.boss)
          "automaton": 3, "collector": 4, "champ": 5}
ROOM = {"$": 0, "R": 1, "?": 2, "E": 3, "M": 4, "T": 5}  # sts::Room order; anything else (N) = NONE


# ------------------------------------------------------------------ reward
def score(run, progress=0.25, hp=0.0):
    """The return of a finished run, in [0, 1].
    Clear: 1 - hp + hp * final_hp / max_hp  (hp = 0: plain 1).  Death on floor f: progress * f / 16."""
    if run["status"] == "act_complete":
        last = run["steps"][-1]["state"]
        return 1.0 - hp + hp * last["hp"] / max(last["max_hp"], 1)
    return progress * min(run["floor"], 16) / 16


# Full-game target ("floors"): linear in floor reached + step for each act boss beaten (act 2's bigger).
# Act 1 boss is floor 16, act 2 boss floor 33 (floors keep counting across acts). A full act-2 clear = 1.
FLOOR_W, ACT_BONUS, LAST_FLOOR = 0.4, (0.2, 0.4), 33


def acts_cleared(run):
    if "acts_cleared" in run:
        return run["acts_cleared"]
    return sum(1 for s in run["steps"] if s["kind"] == "fight" and s.get("category") == "boss" and s["won"])


def floor_score(floor, cleared):
    """Run score in [0, 1] from the floor reached and the number of act bosses beaten."""
    return FLOOR_W * min(floor, LAST_FLOOR) / LAST_FLOOR + sum(ACT_BONUS[:cleared])


def run_score(run, target="act1", progress=0.25, hp=0.0):
    """target act1: the original Act-1 score (score()); floors: floor_score."""
    if target == "act1":
        return score(run, progress, hp)
    return floor_score(run["floor"], acts_cleared(run))


def nodes(run):
    """The run's training nodes: (state, options, column) with column = option index or None for skip."""
    steps = run["steps"]
    last = steps[-1]
    terminal = last["kind"] == "fight" and (not last["won"] or run["status"] == "act_complete")
    out = []
    for s in steps[:-1] if terminal else steps:
        if s["kind"] == "decide":
            if "after" in s:  # neow has none (bandit-decided, agents/overworld/neow.py)
                out.append((s["after"], [], None))
        elif s["kind"] == "pick":
            c = s["choice"]
            out.append((s["state"], s["options"], c if c < len(s["options"]) else None))
        else:
            out.append((s["state"], [], None))
    return out


def node_bosses(run):
    """The current act's boss of each node of nodes(run): the step's own "boss" (multi-act runs), else the run's."""
    steps = run["steps"]
    last = steps[-1]
    terminal = last["kind"] == "fight" and (not last["won"] or run["status"] == "act_complete")
    out = []
    for s in steps[:-1] if terminal else steps:
        if s["kind"] != "decide" or "after" in s:
            out.append(s.get("boss", run["boss"]))
    return out


# ------------------------------------------------------------------ encoding
def encode(items, boss_names, device="cpu"):
    """items: [(state, options)], boss_names: [str] -> run_policy_v1 input dict. Skip column = K (last)."""
    B = len(items)
    C = max(len(s["deck"]) for s, _ in items)
    R = max(max(len(s["relics"]) for s, _ in items), 1)
    P = max(max(len(s["potions"]) for s, _ in items), 1)
    K = max(max(len(o) for _, o in items), 1)
    M = max(max(len(s["map"]["paths"]) for s, _ in items), 1)
    f, i64 = np.float32, np.int64
    card = np.zeros((3, B, C), f); card_mask = np.zeros((B, C), f)
    relic_id = np.zeros((B, R), i64); relic_data = np.zeros((B, R), f); relic_mask = np.zeros((B, R), f)
    potion_id = np.zeros((B, P), i64); potion_mask = np.zeros((B, P), f)
    opt = np.zeros((3, B, K), f); opt_mask = np.zeros((B, K), f)
    boss = np.zeros(B, i64); scalars = np.zeros((B, 6), f)
    path_room = np.full((B, M, 15), NONE, i64); path_mask = np.zeros((B, M), f)
    for i, ((s, opts), name) in enumerate(zip(items, boss_names)):
        d = s["deck"]
        if d:
            card[:, i, :len(d)] = np.array([(c["card_id"], c["upgraded"], c["misc"]) for c in d], f).T
            card_mask[i, :len(d)] = 1
        rl = s["relics"]
        if rl:
            relic_id[i, :len(rl)] = [r["relic_id"] for r in rl]; relic_data[i, :len(rl)] = [r["data"] for r in rl]
            relic_mask[i, :len(rl)] = 1
        po = s["potions"]
        if po:
            potion_id[i, :len(po)] = [q["potion_id"] for q in po]; potion_mask[i, :len(po)] = 1
        if opts:
            opt[:, i, :len(opts)] = np.array([(c["card_id"], c["upgraded"], c["misc"]) for c in opts], f).T
            opt_mask[i, :len(opts)] = 1
        boss[i] = BOSSES[s.get("boss", name)]  # a state's own boss (multi-act worker) wins over the caller's
        hp, mx = s["hp"], max(s["max_hp"], 1)
        scalars[i] = (hp / 100, mx / 100, hp / mx, s["gold"] / 100, s["floor"] / 17, s["potion_capacity"] / 5)
        paths = s["map"]["paths"]
        for j, q in enumerate(paths):
            path_room[i, j] = [ROOM.get(ch, NONE) for ch in q["rooms"]]
        path_mask[i, :max(len(paths), 1)] = 1  # no paths left: one empty path (all NONE) so the max-pool is defined
    t = lambda a: torch.from_numpy(a).to(device)
    return {"card_id": t(card[0].astype(i64)), "card_up": t(card[1]), "card_misc": t(card[2]), "card_mask": t(card_mask),
            "relic_id": t(relic_id), "relic_data": t(relic_data), "relic_mask": t(relic_mask),
            "potion_id": t(potion_id), "potion_mask": t(potion_mask),
            "opt_id": t(opt[0].astype(i64)), "opt_up": t(opt[1]), "opt_misc": t(opt[2]), "opt_mask": t(opt_mask),
            "boss": t(boss), "scalars": t(scalars), "path_room": t(path_room), "path_mask": t(path_mask)}


def columns(options_list, choices, K):
    """Column of each node's value: the option index, or K (skip) for None."""
    return torch.tensor([K if c is None else c for c in choices], dtype=torch.long)


@torch.no_grad()
def node_values(model, run, device="cpu"):
    """V of every node of `run` under `model` (eval mode)."""
    ns = nodes(run)
    if not ns:
        return []
    b = encode([(s, o) for s, o, _ in ns], node_bosses(run), device)
    K = b["opt_id"].shape[1]
    win = model(b)[0]
    col = columns(None, [c for _, _, c in ns], K).to(device)
    return torch.sigmoid(win.gather(1, col[:, None])).squeeze(1).tolist()


@torch.no_grad()
def all_node_values(model, runs, device="cpu", chunk=256):
    """node_values for many runs at once (batched): list of lists."""
    flat = [(s, o, c, b) for r in runs for (s, o, c), b in zip(nodes(r), node_bosses(r))]
    vals = []
    for i in range(0, len(flat), chunk):
        part = flat[i:i + chunk]
        b = encode([(s, o) for s, o, _, _ in part], [x[3] for x in part], device)
        K = b["opt_id"].shape[1]
        col = torch.tensor([K if c is None else c for _, _, c, _ in part], device=device)
        vals += torch.sigmoid(model(b)[0].gather(1, col[:, None])).squeeze(1).tolist()
    out, k = [], 0
    for r in runs:
        n = len(nodes(r))
        out.append(vals[k:k + n]); k += n
    return out


def td_targets(values, g, lam):
    """TD(lambda) targets for nodes with values V_0..V_{n-1}, all reward at the end (g):
    t_{n-1} = g;  t_i = (1 - lam) * V_{i+1} + lam * t_{i+1}."""
    n = len(values)
    t = [0.0] * n
    nxt = g
    for i in reversed(range(n)):
        t[i] = nxt
        nxt = (1 - lam) * values[i] + lam * t[i]
    return t


# ------------------------------------------------------------------ checkpoints / logs
def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = KINDS[ck.get("kind", RunPolicyV1.KIND)](**ck["args"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model


def save_model(model, path, **meta):
    torch.save({"kind": model.KIND, "args": model.args, "state_dict": model.state_dict(), **meta}, path)


def read_runs(paths):
    runs = []
    for p in paths:
        path = Path(p)
        canonical = sorted(path.glob("runs-*.parquet")) if path.is_dir() else []
        if canonical:
            import pyarrow.parquet as pq
            for f in canonical:
                runs += [json.loads(x) for x in pq.ParquetFile(f).read(columns=["record_json"])["record_json"].to_pylist()]
        else:
            for f in sorted(path.glob("**/runs.jsonl")) if path.is_dir() else [path]:
                runs += [json.loads(line) for line in open(f)]
    return runs
