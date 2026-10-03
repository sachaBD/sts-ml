"""Summarise (and pair-compare) Heart-mode runs.  Usage: analyze.py RUNS_DIR [RUNS_DIR2] [--json]"""
import argparse
import json
import math
from collections import Counter, defaultdict

from agents.overworld.value.core import acts_cleared, read_runs, run_score


def keys_of(run):
    for s in reversed(run["steps"]):
        if "overworld" in s.get("state", {}):
            return s["state"]["overworld"].get("keys", {})
    return {}


def metrics(r):
    """Per-run scalar metrics (floats)."""
    c, k = acts_cleared(r), keys_of(r)
    m = {"full_score": run_score(r, "full"), "act1_clear": c >= 1, "act2_clear": c >= 2, "act3_clear": c >= 3,
         "act4_entered": c >= 3 and r["status"] != "heart_locked", "heart_kill": bool(r.get("heart_cleared")),
         "heart_locked": r["status"] == "heart_locked", "floor": r["floor"], "seconds": r.get("seconds", float("nan"))}
    m.update({f"key_{n}": bool(k.get(n, False)) for n in ("ruby", "sapphire", "emerald")})
    return {n: float(v) for n, v in m.items()}


def mean_se(xs):
    xs = [x for x in xs if x == x]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    return mu, (math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1) / n) if n > 1 else float("nan"))


def summarize(runs):
    ms = [metrics(r) for r in runs]
    out = {"n": len(runs)}
    for name in ms[0] if ms else []:
        out[name], out[name + "_se"] = mean_se([m[name] for m in ms])
    deaths, boss = Counter(), defaultdict(list)
    for r in runs:
        fights = [s for s in r["steps"] if s["kind"] == "fight"]
        if fights and not fights[-1]["won"]:
            f = fights[-1]
            deaths[(f.get("state", {}).get("act", r.get("act")), f.get("encounter"))] += 1
        for f in fights:
            if f.get("category") == "boss":
                boss[f["encounter"]].append((f["hp_before"], f["won"]))
    out["deaths"] = deaths.most_common(15)
    out["boss"] = {e: (len(v), sum(h for h, _ in v) / len(v), sum(w for _, w in v) / len(v)) for e, v in boss.items()}
    return out


def show(s, label):
    print(f"== {label}: n={s['n']}")
    if not s["n"]:
        return
    print(f"full score {s['full_score']:.4f} ± {s['full_score_se']:.4f}   mean floor {s['floor']:.1f}   "
          f"worker-s/run {s['seconds']:.1f}")
    for names in (("act1_clear", "act2_clear", "act3_clear", "act4_entered", "heart_kill", "heart_locked"),
                  ("key_ruby", "key_sapphire", "key_emerald")):
        print("  ".join(f"{n} {s[n]:.3f}±{s[n + '_se']:.3f}" for n in names))
    print("boss fights (encounter: n, mean hp_before, win rate):")
    for e, (n, hp, w) in sorted(s["boss"].items()):
        print(f"  {e:20s} n={n:4d} hp={hp:6.1f} win={w:.3f}")
    print("deaths by (act, encounter):")
    for (act, e), c in s["deaths"]:
        print(f"  act{act} {e}: {c}")


def paired(a, b):
    ka, kb = {r["seed"]: r for r in a}, {r["seed"]: r for r in b}
    if set(ka) != set(kb):
        import sys
        common = set(ka) & set(kb)
        print(f"WARNING seed sets differ: {len(ka)} vs {len(kb)}; pairing on {len(common)} common seeds", file=sys.stderr)
        ka = {s: r for s, r in ka.items() if s in common}; kb = {s: r for s, r in kb.items() if s in common}
    out = {}
    for seed in sorted(ka):
        ma, mb = metrics(ka[seed]), metrics(kb[seed])
        for n in ma:
            out.setdefault(n, []).append(mb[n] - ma[n])
    return {n: (*mean_se(d), len(d)) for n, d in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    if len(args.dirs) > 2:
        ap.error("at most two RUNS_DIRs")
    runs = [read_runs([d]) for d in args.dirs]
    sums = [summarize(r) for r in runs]
    diffs = paired(*runs) if len(runs) == 2 else None
    if args.json:
        head = lambda s: {k: v for k, v in s.items() if k not in ("deaths", "boss")}
        print(json.dumps({"runs": [head(s) for s in sums],
                          "paired_b_minus_a": {n: {"diff": d, "se": se, "n": n_} for n, (d, se, n_) in diffs.items()} if diffs else None}))
        return
    for d, s in zip(args.dirs, sums):
        show(s, d)
    if diffs:
        print(f"== paired (B - A), diff ± 1 SE")
        for n, (d, se, n_) in diffs.items():
            print(f"  {n:14s} {d:+.4f} ± {se:.4f}  (n={n_})")


if __name__ == "__main__":
    main()
