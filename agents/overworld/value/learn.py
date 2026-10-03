#!/usr/bin/env python3
"""Train the run value V (run_policy_v1 value head) on logged real runs with TD(lambda) targets.

  train.py --data DIR [DIR ...] --out CKPT [--init CKPT] [--lam 0.7] [--progress 0.25] [--hp 0.0] [--epochs 30]

Targets: each run's nodes (common.nodes) get TD(lambda) targets from the run's score (common.score) and the
values of the NEXT nodes under the --init model; --decay weights older --data dirs less (lam = 1 or no --init: pure Monte Carlo, target = score).
Loss: BCE(sigmoid(win_logit[node column]), target). Split by seed (10% held out) for early stopping.
"""
import argparse
import hashlib
import itertools
import json
import random
import resource
import sys
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as F


from agents.overworld.value.core import all_node_values, build_model, encode, load_model, iter_runs, node_bosses, nodes, save_model, run_score, td_targets, LAST_FLOOR, LAST_FLOOR3, LAST_FLOOR_HEART  # noqa: E402


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
        aux = aux_targets(run, 16 if target == "act1" else LAST_FLOOR_HEART if target in ("heart", "full") else LAST_FLOOR3 if target == "floors3" else LAST_FLOOR)
        ex += [(s, o, c, b, t, run["seed"], weights[k], aux) for (s, o, c), t, b in zip(ns, targets, node_bosses(run))]
    return ex


ORIG = {}  # key -> dtype produced by encode(); compact tensors are cast back to it (lossless) on use


def compact(b):
    """Store an encoded batch in the smallest lossless dtypes: ints -> uint8/int16/int32, 0..255 floats -> uint8,
    half-exact floats -> float16 (each only when the cast round-trips exactly)."""
    out = {}
    for k, v in b.items():
        ORIG.setdefault(k, v.dtype)
        if v.numel() and v.dtype == torch.int64:
            lo, hi = v.min().item(), v.max().item()
            v = v.to(torch.uint8 if 0 <= lo and hi < 256 else torch.int16 if -32768 <= lo and hi < 32768 else torch.int32)
        elif v.numel() and v.dtype == torch.float32:
            u = v.to(torch.uint8)
            if torch.equal(u.float(), v):
                v = u
            else:
                h = v.half()
                v = h if torch.equal(h.float(), v) else v
        out[k] = v
    return out


def to(b, device):
    return {k: v.to(device, non_blocking=True).to(ORIG[k]) for k, v in b.items()}


TEACHER = None  # optional model whose values on EVERY column (all offered options + skip) are distillation targets


def teacher_targets(chunk, K):
    """Dense distillation targets in the student's column layout [B, K+1] (options 0..n-1, skip at K) + mask."""
    with torch.no_grad():
        v = torch.sigmoid(TEACHER(encode([(s, o) for s, o, *_ in chunk], [e[3] for e in chunk], kind=TEACHER.KIND))[0]).cpu()
    D, M = torch.zeros(len(chunk), K + 1), torch.zeros(len(chunk), K + 1, dtype=torch.bool)
    for i, e in enumerate(chunk):
        n = len(e[1])
        D[i, :n], M[i, :n] = v[i, :n], True
        D[i, K], M[i, K] = v[i, -1], True
    return D.half(), M


def encode_batch(chunk, kind):
    """chunk of examples -> (compact batch, column, target, weight, aux[, teacher targets, mask])."""
    b = encode([(s, o) for s, o, *_ in chunk], [e[3] for e in chunk], kind=kind)
    K = b["opt_id"].shape[1]
    col = torch.tensor([K if e[2] is None else e[2] for e in chunk])
    out = (compact(b), col, torch.tensor([e[4] for e in chunk]), torch.tensor([e[6] for e in chunk]),
           torch.tensor([e[7] for e in chunk]))
    return out + teacher_targets(chunk, K) if TEACHER is not None else out


def batches(ex, size, device, shuffle, kind=None):
    """Encoded minibatches (b, column, target, weight, aux), kept on the CPU in compact dtypes (moved per step: all of
    them on the GPU overflow its memory); encode once, reuse every epoch."""
    idx = list(range(len(ex)))
    if shuffle:
        random.shuffle(idx)
    return [encode_batch([ex[j] for j in idx[i:i + size]], kind) for i in range(0, len(idx), size)]


def stream_batches(path_groups, weights_of, init, lam, progress, hp, device, target, kind, train_size, val_size, group=96):
    """One pass over the runs, `group` runs at a time (so the parsed JSON never accumulates): targets -> examples ->
    held-out split (as always: by seed) -> train nodes shuffled within the group -> compact encoded batches.
    Returns (train batches, val batches, run count, mean score, any heart kill)."""
    train, val, tp, vp = [], [], [], []
    stats = {"runs": 0, "score": 0.0, "heart": False}

    def process(grp, ws, final=False):
        nonlocal tp, vp
        ex = build_examples(grp, ws, init, lam, progress, hp, device, target)
        tp += [e for e in ex if not is_held(e[5])]
        vp += [e for e in ex if is_held(e[5])]
        for r in grp:
            stats["runs"] += 1
            stats["score"] += run_score(r, target, progress, hp)
            stats["heart"] |= bool(r.get("heart_cleared", False))
        keep_t, keep_v = [], []
        for pend, size, dst, shuf, keep in ((tp, train_size, train, True, keep_t), (vp, val_size, val, False, keep_v)):
            if shuf:
                random.shuffle(pend)
            while len(pend) >= size or (final and pend):
                dst.append(encode_batch(pend[:size], kind))
                del pend[:size]
            keep += pend  # < one batch of leftovers carried to the next group
        tp, vp = keep_t, keep_v

    grp, ws = [], []
    for k, run in path_groups:
        grp.append(run); ws.append(weights_of(k))
        if len(grp) >= group:
            process(grp, ws); grp, ws = [], []
    process(grp, ws, final=True)
    return train, val, stats["runs"], stats["score"] / max(stats["runs"], 1), stats["heart"]


def is_held(seed):
    return random.Random(seed).random() < 0.1


def evaluate(model, val, device, distill_weight=0.0):
    """Mean val BCE of the chosen column; with distillation targets, + distill_weight * the per-node mean dense
    distillation BCE (early stopping then tracks the full training objective)."""
    model.eval()
    tot, dtot, n = 0.0, 0.0, 0
    with torch.no_grad():
        for b, col, y, _, _, *dist in val:
            col, y = col.to(device), y.to(device)
            out = model(to(b, device))[0]
            logit = out.gather(1, col[:, None]).squeeze(1)
            tot += F.binary_cross_entropy_with_logits(logit, y, reduction="sum").item()
            if dist and distill_weight:
                D, M = dist[0].to(device).float(), dist[1].to(device)
                dtot += F.binary_cross_entropy_with_logits(out[M], D[M]).item() * len(y)
            n += len(y)
    return (tot + distill_weight * dtot) / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init")
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--progress", type=float, default=0.25)
    ap.add_argument("--hp", type=float, default=0.0)
    ap.add_argument("--target", choices=["act1", "floors", "floors3", "full", "heart"], default="act1",
                    help="act1/floors/floors3/full: progress targets (full = Heart-mode progress); heart: binary actual Heart defeat (max-act 4)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, help="default 32 for graph models, 256 for legacy")
    ap.add_argument("--val-batch-size", type=int, help="default 64 for graph models, 512 for legacy")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--td-model", help="model giving the TD bootstrap values (default: --init)")
    ap.add_argument("--distill", help="teacher checkpoint: its values on every offered option + skip are extra targets")
    ap.add_argument("--distill-weight", type=float, default=1.0)
    ap.add_argument("--aux-weight", type=float, default=0.5, help="weight of auxiliary targets (models with aux > 0)")
    arch_group = ap.add_mutually_exclusive_group()
    arch_group.add_argument("--arch", default="{}", help='new model (no --init): {"kind": ..., args}; default run_policy_v1')
    arch_group.add_argument("--arch-spec", type=Path, help="named architecture TOML (kind + [args]); new model only")
    ap.add_argument("--max-runs", type=int, help="use only the first N runs (smoke tests / memory measurement)")
    ap.add_argument("--encode-only", action="store_true", help="encode the data, print node counts and peak RSS, stop")
    ap.add_argument("--threads", type=int, help="torch CPU threads (default: torch's)")
    ap.add_argument("--decay", type=float, default=1.0,
                    help="loss weight of a --data dir = decay ** (dirs after it); 1 = all equal")
    a = ap.parse_args()
    if a.arch_spec and a.init:
        ap.error("--arch-spec builds a new model and cannot be combined with --init")
    if a.epochs < 1 or any(n is not None and n < 1 for n in (a.batch_size, a.val_batch_size)):
        ap.error("epochs and batch sizes must be positive")
    architecture = json.loads(a.arch)
    spec_sha = None
    if a.arch_spec:
        spec_sha = hashlib.sha256(a.arch_spec.read_bytes()).hexdigest()
        spec = tomllib.loads(a.arch_spec.read_text())
        architecture = {"kind": spec["kind"], **spec["args"]}
        frozen = a.arch_spec.parent / "FROZEN.tsv"
        if frozen.exists():
            for line in frozen.read_text().splitlines():
                name, sha, *_ = line.split("\t")
                if name == a.arch_spec.stem and sha != spec_sha:
                    ap.error(f"frozen architecture changed: {a.arch_spec}")
    random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.threads:
        torch.set_num_threads(a.threads)
    init = load_model(a.init, device) if a.init else None
    td = load_model(a.td_model, device) if a.td_model else init
    if a.distill:
        global TEACHER
        TEACHER = load_model(a.distill, "cpu").eval()
    kind = init.KIND if init else architecture.get("kind", "run_policy_v1")
    graph = kind in ("run_policy_v3_1", "run_policy_v3_2")
    stream = iter_runs(a.data)
    if a.max_runs:
        stream = itertools.islice(stream, a.max_runs)
    train, val, n_runs, mean_score, heart = stream_batches(
        stream, lambda k: a.decay ** (len(a.data) - 1 - k), td if a.lam < 1 else None, a.lam, a.progress, a.hp, device,
        a.target, kind, a.batch_size or (32 if graph else 256), a.val_batch_size or (64 if graph else 512))
    if not n_runs:
        ap.error("no runs found in --data")
    if a.target == "heart" and not heart:
        print("WARNING: no positive Heart outcomes; consider progress pretraining or better collection before binary Heart training", file=sys.stderr)
    print(f"{n_runs} runs, {sum(len(b[1]) for b in train)} train nodes, {sum(len(b[1]) for b in val)} val nodes; "
          f"mean score {mean_score:.3f}", flush=True)
    if a.encode_only:
        print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB; "
              f"encoded {sum(v.numel() * v.element_size() for b in train + val for v in b[0].values()) / 1e6:.0f} MB")
        return
    model = (build_model({"kind": init.KIND, **init.args}) if init else build_model(architecture)).to(device)
    if init:
        model.load_state_dict(init.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    best, best_state, bad = float("inf"), None, 0
    base = evaluate(model, val, device, a.distill_weight if a.distill else 0.0)
    print(f"epoch 0 val {base:.4f}", flush=True)
    for epoch in range(1, a.epochs + 1):
        model.train()
        random.shuffle(train)
        for b, col, y, w, aux, *dist in train:
            col, y, w = col.to(device), y.to(device), w.to(device)
            out = model(to(b, device))
            logit = out[0].gather(1, col[:, None]).squeeze(1)
            loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / w.sum()
            if dist:  # dense distillation: every offered option + skip toward the teacher's value
                D, M = dist[0].to(device).float(), dist[1].to(device)
                loss = loss + a.distill_weight * F.binary_cross_entropy_with_logits(out[0][M], D[M])
            if getattr(model, "aux_out", None) is not None:  # run_policy_v2 aux head only (v1's extra heads are untrained)
                al = out[1].gather(1, col[:, None, None].expand(-1, 1, out[1].shape[-1])).squeeze(1)
                lx = F.binary_cross_entropy_with_logits(al, aux.to(device), reduction="none").mean(1)
                loss = loss + a.aux_weight * (lx * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        v = evaluate(model, val, device, a.distill_weight if a.distill else 0.0)
        print(f"epoch {epoch} val {v:.4f}", flush=True)
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(best_state)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_model(model.cpu(), a.out, data=a.data, target=a.target, lam=a.lam, progress=a.progress, hp=a.hp, decay=a.decay, val_bce=best,
               runs=n_runs, architecture_spec=str(a.arch_spec) if a.arch_spec else None, architecture_spec_sha256=spec_sha)
    if a.arch_spec:
        frozen = a.arch_spec.parent / "FROZEN.tsv"
        previous = frozen.read_text() if frozen.exists() else ""
        if not any(line.split("\t")[0] == a.arch_spec.stem for line in previous.splitlines()):
            with frozen.open("a") as f:
                f.write(f"{a.arch_spec.stem}\t{spec_sha}\t{a.out}\n")
    print(json.dumps({"out": a.out, "val_bce": best, "runs": n_runs}))


if __name__ == "__main__":
    main()
