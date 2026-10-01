#!/usr/bin/env python3
"""elite-bench analysis: headroom (oracle), A/A noise floor, v3 vs MCTS with per-fight win probabilities, budget
and particle curves, power. Works on whatever runs exist so far (sections without their runs are skipped).

    PYTHONPATH=python .venv/bin/python experiments/elite-bench/analyze.py [--prefix bench] [--store runs]
        [--output experiments/elite-bench/results/analysis.md]

All comparisons are paired on identical fight starts (same stored fight, same game RNG). A fair arm's per-fight
score is its mean over the salts it was played with (an unbiased estimate of that fight's expected HP-eq / win
probability); pivotal fights carry extra salts (S5), which only lowers their variance. CIs: 95% percentile,
10,000 bootstraps over source run seeds.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from bench import ELITES, arms, cluster_ci, fmt, load

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="bench")
p.add_argument("--store", default="runs")
p.add_argument("--output", default=str(Path(__file__).parent / "results" / "analysis.md"))
a = p.parse_args()

A = arms(load(a.prefix, a.store))
rng = np.random.default_rng(2026)
L: list[str] = []
J: dict = {}
out = L.append


def table(header, rows):
    out("| " + " | ".join(header) + " |")
    out("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
    for r in rows:
        out("| " + " | ".join(str(x) for x in r) + " |")
    out("")


def per_fight(arm, eps=None, salts=None):
    """{ep: (mean hpeq, win prob, n salts, fight)} over the arm's salts that played ep."""
    runs = A.get(arm, {})
    res = {}
    for s, fights in runs.items():
        if salts is not None and s not in salts:
            continue
        for ep, f in fights.items():
            if eps is not None and ep not in eps:
                continue
            res.setdefault(ep, []).append(f)
    return {ep: (np.mean([f["hpeq"] for f in fs]), np.mean([f["won"] for f in fs]), len(fs), fs[0])
            for ep, fs in res.items()}


def common(*dicts):
    keys = set(dicts[0])
    for d in dicts[1:]:
        keys &= set(d)
    return sorted(keys)


def by_elite(eps, meta):
    yield "all", eps
    for e in ELITES:
        yield e, [x for x in eps if meta[x]["enc"] == e]


out(f"# elite-bench analysis ({a.prefix})")
out("")
out("Runs found: " + ", ".join(f"{arm} (salts {sorted(s)}; {len(next(iter(s.values())))} fights)"
                               for arm, s in sorted(A.items())))
out("")

# ---------------------------------------------------------------- A. headroom and fight classes
M20 = per_fight("mcts-20k", salts=range(4))
V20 = per_fight("v3-20k", salts=range(4))
ORA = per_fight("oracle-20k")
# Main estimates use salts 0-3 on every fight. Pivotal fights were chosen from salts 0-3 outcomes, so their extra
# salts 4-7 (S5) are reported separately (section C2): a selection-independent, held-out estimate on that stratum.
M20all, V20all = M20, V20
meta = {ep: v[3] for d in (M20, V20, ORA) for ep, v in d.items()}
if ORA and M20:
    out("## A. Headroom: oracle (sees draws and RNG) vs fair play, 20k simulations")
    out("")
    out("Fair columns: mean over salts 0-3 (per-fight expected value). Unwinnable = the oracle lost. "
        "Pivotal = some fair 20k run lost and the fight is winnable (oracle or some fair run won). "
        "HP-eq lost = 35 + starting HP - HP-eq (the most a perfect player could save).")
    out("")
    eps = common(ORA, M20, V20) if V20 else common(ORA, M20)
    rows, J["headroom"] = [], {}
    for name, sub in by_elite(eps, meta):
        n = len(sub)
        fair_runs = [M20] + ([V20] if V20 else [])
        unwin = [x for x in sub if not ORA[x][1]]
        piv = [x for x in sub if min(d[x][1] for d in fair_runs) < 1 and (ORA[x][1] or max(d[x][1] for d in fair_runs) > 0)]
        lost = lambda d: np.mean([35 + meta[x]["hp"] - d[x][0] for x in sub])
        row = [name, n, f"{np.mean([ORA[x][1] for x in sub]):.1%}", f"{np.mean([M20[x][1] for x in sub]):.1%}",
               f"{np.mean([V20[x][1] for x in sub]):.1%}" if V20 else "-",
               f"{lost(ORA):.1f}", f"{lost(M20):.1f}", f"{lost(V20):.1f}" if V20 else "-",
               f"{len(unwin)} ({len(unwin)/n:.1%})", f"{len(piv)} ({len(piv)/n:.1%})",
               sum(1 for x in unwin if M20[x][1] > 0)]
        rows.append(row)
        J["headroom"][name] = dict(n=n, oracle_win=float(np.mean([ORA[x][1] for x in sub])),
                                   mcts_win=float(np.mean([M20[x][1] for x in sub])),
                                   oracle_lost=float(lost(ORA)), mcts_lost=float(lost(M20)),
                                   unwinnable=len(unwin), pivotal=len(piv))
    table(["elite", "fights", "win oracle", "win MCTS", "win v3", "HP-eq lost oracle", "HP-eq lost MCTS",
           "HP-eq lost v3", "unwinnable", "pivotal", "unwinnable but MCTS won (sanity)"], rows)
    gap = [ORA[x][0] - M20[x][0] for x in eps]
    out(f"Oracle − MCTS HP-eq per fight (upper bound on fair headroom; includes the value of seeing the draws): "
        f"{fmt(cluster_ci(gap, [meta[x]['seed'] for x in eps], rng))}")
    out("")
    O50 = per_fight("oracle-50k")
    if O50:
        c = common(O50, ORA)
        d = [O50[x][0] - ORA[x][0] for x in c]
        out(f"Oracle convergence (subset, n={len(c)}): 50k − 20k = {fmt(cluster_ci(d, [meta[x]['seed'] for x in c], rng))} HP-eq, "
            f"wins {sum(ORA[x][1] for x in c):.0f} → {sum(O50[x][1] for x in c):.0f}")
        out("")

# ---------------------------------------------------------------- B. A/A noise floor
def pair_stats(r1, r2):
    c = common(r1, r2)
    d = np.array([r2[x]["hpeq"] - r1[x]["hpeq"] for x in c])
    return dict(n=len(c), mean=float(d.mean()), sd=float(d.std()),
                identical=int((np.abs(d) < 1e-6).sum()),
                only1=sum(r1[x]["won"] and not r2[x]["won"] for x in c),
                only2=sum(r2[x]["won"] and not r1[x]["won"] for x in c), c=c, d=d)


if len(A.get("mcts-20k", {})) >= 2:
    out("## B. Noise floor: the same agent under different search salts (A/A)")
    out("")
    out("Each row: two runs of the listed arms on identical fights. 'discordant wins' = fights won by only one "
        "run (first / second). An A/A row shows how much of a single-run comparison is pure chance.")
    out("")
    rows, J["aa"] = [], []
    pairs = [("mcts-20k", s1, "mcts-20k", s2) for s1, s2 in itertools.combinations(sorted(s for s in A["mcts-20k"] if s < 4), 2)]
    if "v3-20k" in A:
        pairs += [("v3-20k", s1, "v3-20k", s2) for s1, s2 in itertools.combinations(sorted(s for s in A["v3-20k"] if s < 4), 2)]
        pairs += [("mcts-20k", s, "v3-20k", s) for s in sorted(set(A["mcts-20k"]) & set(A["v3-20k"])) if s < 4]
    for a1, s1, a2, s2 in pairs:
        st = pair_stats(A[a1][s1], A[a2][s2])
        rows.append([f"{a1} s{s1} vs {a2} s{s2}", st["n"], f"{st['mean']:+.2f}", f"{st['sd']:.1f}",
                     f"{st['identical']/st['n']:.0%}", f"{st['only1']} / {st['only2']}"])
        J["aa"].append({k: v for k, v in st.items() if k not in ("c", "d")} | dict(a1=a1, s1=s1, a2=a2, s2=s2))
    table(["pair", "fights", "mean diff", "SD of diff", "identical score", "discordant wins"], rows)
    # per-fight variance decomposition (MCTS, salts 0-3)
    runs = [A["mcts-20k"][s] for s in sorted(A["mcts-20k"]) if s < 4]
    c = common(*runs)
    X = np.array([[r[x]["hpeq"] for r in runs] for x in c])
    W = np.array([[r[x]["won"] for r in runs] for x in c])
    within, total = X.var(axis=1, ddof=1).mean(), X.var(ddof=1)
    varies = (W.min(axis=1) != W.max(axis=1))
    out(f"MCTS 20k, {len(runs)} salts, {len(c)} fights: within-fight (salt) variance {within:.0f} of total {total:.0f} "
        f"HP-eq² ({within/total:.0%}); outcome (win/loss) varies across salts in {varies.sum()} fights ({varies.mean():.1%}). "
        f"Fights never won: {(W.max(axis=1)==0).sum()}, always won: {(W.min(axis=1)==1).sum()}.")
    out("")
    J["variance"] = dict(within=float(within), total=float(total), outcome_varies=int(varies.sum()), fights=len(c))

# ---------------------------------------------------------------- C. v3 vs MCTS with K salts
if M20 and V20:
    out("## C. v3 (policy priors) vs MCTS, 20k simulations: per-fight means over salts")
    out("")
    out("Δ = v3 − MCTS HP-eq per fight. 'K=1' uses salt 0 only (the old protocol); 'all salts' uses salts 0-3 "
        "on every fight. Winnable = the oracle won.")
    out("")
    eps = common(M20all, V20all)
    K1m, K1v = per_fight("mcts-20k", salts={0}), per_fight("v3-20k", salts={0})
    rows, J["v3_vs_mcts"] = [], {}
    for name, sub in by_elite(eps, meta):
        seeds = [meta[x]["seed"] for x in sub]
        d_all = [V20all[x][0] - M20all[x][0] for x in sub]
        d_k1 = [K1v[x][0] - K1m[x][0] for x in sub if x in K1v and x in K1m]
        s_k1 = [meta[x]["seed"] for x in sub if x in K1v and x in K1m]
        win = [x for x in sub if x in ORA and ORA[x][1]]
        d_win = [V20all[x][0] - M20all[x][0] for x in win]
        ci_all = cluster_ci(d_all, seeds, rng)
        rows.append([name, len(sub), fmt(cluster_ci(d_k1, s_k1, rng)), fmt(ci_all),
                     fmt(cluster_ci(d_win, [meta[x]["seed"] for x in win], rng)) if win else "-",
                     f"{np.mean([M20all[x][1] for x in sub]):.1%} / {np.mean([V20all[x][1] for x in sub]):.1%}"])
        J["v3_vs_mcts"][name] = dict(n=len(sub), mean=ci_all[0], lo=ci_all[1], hi=ci_all[2])
    table(["elite", "fights", "Δ HP-eq, K=1", "Δ HP-eq, all salts", "Δ HP-eq, winnable only", "win prob MCTS / v3"], rows)
    # long vs short Sentries (length = MCTS salt-0 turns)
    K1 = A["mcts-20k"][0]
    for enc in ("three_sentries", "lagavulin"):
        sub = [x for x in eps if meta[x]["enc"] == enc and x in K1]
        long_ = [x for x in sub if K1[x]["turns"] > 6]
        short = [x for x in sub if K1[x]["turns"] <= 6]
        for lab, s in (("long (>6 turns, MCTS s0)", long_), ("short", short)):
            if s:
                out(f"- {enc} {lab}: n={len(s)}, Δ {fmt(cluster_ci([V20all[x][0]-M20all[x][0] for x in s], [meta[x]['seed'] for x in s], rng))}")
    out("")

# ---------------------------------------------------------------- C2. pivotal stratum, held-out salts
M47, V47 = per_fight("mcts-20k", salts=range(4, 8)), per_fight("v3-20k", salts=range(4, 8))
if M47 and V47:
    out("## C2. Pivotal fights on held-out salts 4-7 (chosen from salts 0-3; no selection bias)")
    out("")
    eps = common(M47, V47)
    rows, J["pivotal_heldout"] = [], {}
    for name, sub in by_elite(eps, meta):
        seeds = [meta[x]["seed"] for x in sub]
        ci = cluster_ci([V47[x][0] - M47[x][0] for x in sub], seeds, rng)
        rows.append([name, len(sub), fmt(ci),
                     f"{np.mean([M20[x][1] for x in sub]):.1%} → {np.mean([M47[x][1] for x in sub]):.1%}",
                     f"{np.mean([V20[x][1] for x in sub]):.1%} → {np.mean([V47[x][1] for x in sub]):.1%}",
                     f"{np.std([V47[x][0] - M47[x][0] for x in sub]):.1f}"])
        J["pivotal_heldout"][name] = dict(n=len(sub), mean=ci[0], lo=ci[1], hi=ci[2])
    table(["elite", "fights", "Δ v3 − MCTS (salts 4-7)", "MCTS win prob salts 0-3 → 4-7", "v3 win prob 0-3 → 4-7",
           "SD of per-fight Δ"], rows)
    out("The drop from salts 0-3 to 4-7 is regression to the mean: how much of 'pivotal' was luck in the selection runs.")
    out("")

# ---------------------------------------------------------------- D. budget curves
out("## D. Budget curves (paired, vs MCTS 20k on the same fights)")
out("")
out("Mean over available salts per fight (MCTS 20k: all its salts). 5k/50k-v3 and oracle-50k arms ran on the 600-fight subset only; compare rows with the same fight count.")
out("")
rows, J["budget"] = [], []
for arm in sorted(A, key=lambda s: (s.split("-")[0], int(s.split("-")[1].rstrip("k")) if s.split("-")[1].rstrip("k").isdigit() else 0)):
    if arm.startswith(("oracle", "r2")) or "-p" in arm or arm in ("mcts-20k", "v3-20k"):
        continue
    R = per_fight(arm, salts=range(4))
    ref = M20all
    c = common(R, ref)
    if not c:
        continue
    for name, sub in by_elite(c, meta):
        d = [R[x][0] - ref[x][0] for x in sub]
        ci = cluster_ci(d, [meta[x]["seed"] for x in sub], rng)
        rows.append([arm, name, len(sub), sorted(A[arm]), fmt(ci),
                     f"{np.mean([ref[x][1] for x in sub]):.1%} → {np.mean([R[x][1] for x in sub]):.1%}"])
        J["budget"].append(dict(arm=arm, elite=name, n=len(sub), mean=ci[0], lo=ci[1], hi=ci[2]))
table(["arm", "elite", "fights", "salts", "Δ HP-eq vs MCTS 20k", "win prob MCTS 20k → arm"], rows)

# v3 vs MCTS at equal budget on the subset
eq = [(b, per_fight(f"mcts-{b}"), per_fight(f"v3-{b}")) for b in ("5k", "20k", "50k")]
eq = [(b, m, v) for b, m, v in eq if m and v]
if len(eq) > 1:
    out("v3 − MCTS at equal budget (subset fights where both budgets exist):")
    out("")
    sub_eps = set.intersection(*[set(common(m, v)) for _, m, v in eq])
    rows = []
    for b, m, v in eq:
        s = sorted(sub_eps)
        rows.append([b, len(s), fmt(cluster_ci([v[x][0] - m[x][0] for x in s], [meta[x]["seed"] for x in s], rng)),
                     f"{np.mean([m[x][0] for x in s]):.1f} / {np.mean([v[x][0] for x in s]):.1f}"])
    table(["budget", "fights", "Δ v3 − MCTS", "mean HP-eq MCTS / v3"], rows)

# ---------------------------------------------------------------- E. particles
P32 = per_fight("mcts-20k-p32")
if P32:
    P8 = per_fight("mcts-20k", salts=range(4, 8))  # held-out salts: pivotal fights were selected on salts 0-3
    c = common(P32, P8)
    out("## E. Particles: MCTS 20k with 32 (salts 0-3) vs 8 particles (held-out salts 4-7), pivotal fights")
    out("")
    rows = []
    for name, sub in by_elite(c, meta):
        rows.append([name, len(sub), fmt(cluster_ci([P32[x][0] - P8[x][0] for x in sub], [meta[x]["seed"] for x in sub], rng)),
                     f"{np.mean([P8[x][1] for x in sub]):.1%} → {np.mean([P32[x][1] for x in sub]):.1%}"])
    table(["elite", "fights", "Δ HP-eq 32 − 8 particles", "win prob 8 → 32"], rows)

# ---------------------------------------------------------------- G. round-2 candidates
cands = sorted(arm for arm in A if arm.startswith("r2"))
if cands and V20 and M20:
    out("## G. Round-2 candidate nets vs elite-v3-t2 (v3-20k) and vs MCTS, 20k, policy priors")
    out("")
    out("vs v3: the same salts on both sides (paired seeds). vs MCTS: candidate's salts vs MCTS salts 0-3.")
    out("")
    rows, J["candidates"] = [], {}
    for arm in cands:
        salts = sorted(s for s in A[arm] if s < 4)
        C = per_fight(arm, salts=set(salts))
        Vs = per_fight("v3-20k", salts=set(salts))
        for name, sub in by_elite(common(C, Vs, M20), meta):
            seeds = [meta[x]["seed"] for x in sub]
            cv = cluster_ci([C[x][0] - Vs[x][0] for x in sub], seeds, rng)
            cm = cluster_ci([C[x][0] - M20[x][0] for x in sub], seeds, rng)
            rows.append([arm, name, len(sub), salts, fmt(cv), fmt(cm),
                         f"{np.mean([M20[x][1] for x in sub]):.1%} / {np.mean([Vs[x][1] for x in sub]):.1%} / {np.mean([C[x][1] for x in sub]):.1%}"])
            J["candidates"][f"{arm}/{name}"] = dict(n=len(sub), vs_v3=cv, vs_mcts=cm)
    table(["arm", "elite", "fights", "salts", "Δ vs v3-t2 (same salts)", "Δ vs MCTS (salts 0-3)",
           "win prob MCTS / v3-t2 / arm"], rows)
    if M47 and V47:
        out("Pivotal fights vs MCTS and v3-t2 on held-out salts 4-7. The candidate uses its own salts 4-7 where it has "
            "them (clean); otherwise its salts 0-3, which it shares with the runs that selected these fights (same "
            "root particles): a slight bias against it. Column 'cand. salts' says which.")
        out("")
        rows = []
        for arm in cands:
            held = {s for s in A[arm] if 4 <= s < 8}
            cs = held or {s for s in A[arm] if s < 4}
            C = per_fight(arm, salts=cs)
            for name, sub in by_elite(common(C, M47, V47), meta):
                seeds = [meta[x]["seed"] for x in sub]
                rows.append([arm, sorted(cs), name, len(sub), fmt(cluster_ci([C[x][0] - M47[x][0] for x in sub], seeds, rng)),
                             fmt(cluster_ci([C[x][0] - V47[x][0] for x in sub], seeds, rng)),
                             f"{np.mean([M47[x][1] for x in sub]):.1%} / {np.mean([V47[x][1] for x in sub]):.1%} / {np.mean([C[x][1] for x in sub]):.1%}"])
        table(["arm", "cand. salts", "elite", "fights", "Δ vs MCTS (4-7)", "Δ vs v3-t2 (4-7)", "win prob MCTS / v3-t2 / arm"], rows)

# ---------------------------------------------------------------- F. power
if M20 and V20:
    out("## F. Power: paired fights needed to detect a true difference (80% power, two-sided 5%)")
    out("")
    eps = common(M20all, V20all)
    K1m, K1v = per_fight("mcts-20k", salts={0}), per_fight("v3-20k", salts={0})
    sd1 = np.std([K1v[x][0] - K1m[x][0] for x in eps if x in K1m and x in K1v])
    sdk = np.std([V20all[x][0] - M20all[x][0] for x in eps])
    rows = []
    for lab, sd in (("K=1 salt per fight", sd1), ("all salts per fight", sdk)):
        rows.append([lab, f"{sd:.1f}", *(f"{int(np.ceil((2.8 * sd / e) ** 2))}" for e in (0.5, 1.0, 2.0))])
    table(["protocol", "SD of per-fight Δ", "n for 0.5 HP-eq", "n for 1.0", "n for 2.0"], rows)
    J["power"] = dict(sd_k1=float(sd1), sd_all=float(sdk))

Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text("\n".join(L) + "\n")
Path(a.output).with_suffix(".json").write_text(json.dumps(J, indent=2, default=float) + "\n")
print("\n".join(L))
