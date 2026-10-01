"""Paired comparison of players (runs) on the same floor-1 easy fights.

  PYTHONPATH=. .venv/bin/python experiments/act-1-easy-combats/analyze.py PREFIX [--ref RUN] [--base RUN]

PREFIX: run id prefix, e.g. easy-main- (all combat_v3 runs whose id starts with it, any date).
Score per fight in HP-eq: final HP + 4 * potions if won, else -35 (the repo's terminal_value, in HP).
Each run is compared with --ref (default: the oracle run with the most simulations) on the fights both played.
Pooled = the mean of the 4 encounter means (the easy pool draws each encounter equally often).
CI: 95%, normal approx on the paired differences, encounters combined as independent strata.
"""
import argparse
import math
import re

from runs import query

ENCOUNTERS = ("cultist", "jaw_worm", "two_louse", "small_slimes")


def load(prefix):
    rows = query.rows(
        f"select * from combat_v3 where schema = 'combat_v3' and id like '{prefix}%' and row_kind = 'decision'"
        " and decision_index = 0",
        ["id", "episode_id", "encounter", "won", "final_hp", "potions", "starting_hp"], None, oracle=True)
    runs = {}
    for r in rows:
        score = r["final_hp"] + 4 * r["potions"] if r["won"] else -35
        runs.setdefault(r["id"], {})[r["episode_id"]] = (r["encounter"], score, r["starting_hp"] - r["final_hp"])
    return runs


def stats(diffs):
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else float("nan")
    return n, mean, var / n, math.sqrt(var)


def compare(a, b):
    """a - b per encounter and pooled: {enc: (n, mean, se, sd, wins a>b, b>a)}."""
    out = {}
    for enc in ENCOUNTERS:
        common = [e for e in a if e in b and a[e][0] == enc]
        if len(common) < 2:
            continue
        diffs = [a[e][1] - b[e][1] for e in common]
        n, mean, var_mean, sd = stats(diffs)
        out[enc] = (n, mean, math.sqrt(var_mean), sd, sum(d > 0 for d in diffs), sum(d < 0 for d in diffs))
    if len(out) == len(ENCOUNTERS):
        mean = sum(v[1] for v in out.values()) / 4
        se = math.sqrt(sum(v[2] ** 2 for v in out.values())) / 4
        out["pooled"] = (sum(v[0] for v in out.values()), mean, se, float("nan"),
                         sum(v[4] for v in out.values()), sum(v[5] for v in out.values()))
    return out


def sims(run_id):
    """Sort key: (oracle, value net, simulations in k, particles), parsed from the run id."""
    m = re.search(r"(fair|oracle|gen1net)(\d+)k(?:-p(\d+))?", run_id)
    return (m.group(1) == "oracle", m.group(1) == "gen1net", int(m.group(2)), int(m.group(3) or 8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prefix")
    p.add_argument("--ref")
    args = p.parse_args()
    runs = load(args.prefix)
    order = sorted(runs, key=sims)
    ref = args.ref or [r for r in order if sims(r)[0]][-1]
    print(f"## {args.prefix}: mean HP lost per fight (all fights played)\n")
    print("| run | " + " | ".join(ENCOUNTERS) + " | fights | deaths |\n|---|" + "---:|" * 6)
    for r in order:
        f = runs[r]
        cells = [f"{sum(v[2] for v in f.values() if v[0] == e) / max(1, sum(v[0] == e for v in f.values())):.2f}"
                 for e in ENCOUNTERS]
        print(f"| {r} | " + " | ".join(cells) + f" | {len(f)} | {sum(v[1] == -35 for v in f.values())} |")
    print(f"\n## run minus {ref}, HP-eq per fight: mean [95% CI] (sd of paired diff; fights better/worse)\n")
    print("| run | " + " | ".join(ENCOUNTERS) + " | pooled |\n|---|" + "---|" * 5)
    for r in order:
        if r == ref:
            continue
        c = compare(runs[r], runs[ref])
        cell = lambda k: (f"{c[k][1]:+.2f} [{c[k][1] - 1.96 * c[k][2]:+.2f}, {c[k][1] + 1.96 * c[k][2]:+.2f}]"
                          + (f" (sd {c[k][3]:.1f}; {c[k][4]}/{c[k][5]})" if k != "pooled" else f" ({c[k][4]}/{c[k][5]})")
                          if k in c else "")
        print(f"| {r} | " + " | ".join(cell(k) for k in (*ENCOUNTERS, "pooled")) + " |")


if __name__ == "__main__":
    main()
