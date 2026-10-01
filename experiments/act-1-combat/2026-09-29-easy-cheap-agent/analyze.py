#!/usr/bin/env python3
"""Easy cheap-agent analysis: quality (paired HP-eq vs MCTS 20k) against cost (search seconds per fight) for every
arm found, plus charts. Works on whatever runs exist so far.

    PYTHONPATH=. .venv/bin/python experiments/act-1-combat/2026-09-29-easy-cheap-agent/analyze.py \
        [--prefix easy-cheap] [--fights fights.csv] [--out results]

Per fight, an arm's score is its mean HP-eq over the seeds (salts) it was played with. HP-eq = terminal_value x
(55 + max HP) = won x (35 + final HP + 4 per potion). Deltas are paired on fights every compared arm played.
Pooled = the 4 encounters weighted equally. CIs: 95% percentile, 10,000 bootstraps over source run seeds.
Cost: search_seconds (move choosing only, inside the worker; 11 workers on a 6-core/12-thread Ryzen 3600X).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
from scipy.stats import gaussian_kde

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
ENCS = ("cultist", "jaw_worm", "small_slimes", "two_louse")
REF = "mcts-20k"
ID_RE = re.compile(r"^(?P<arm>.+?)(?:-s(?P<salt>\d+))?$")

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="easy-cheap")
p.add_argument("--fights", default=str(HERE / "fights.csv"))
p.add_argument("--out", default=str(HERE / "results" / "hpeq"))
p.add_argument("--exclude", default="pilot,confirm", help="skip runs whose arm starts with one of these (comma list)")
a = p.parse_args()
OUT = Path(a.out)
EXCLUDE = tuple(x for x in a.exclude.split(",") if x) or ("\0",)
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2026)

# ------------------------------------------------------------------ load
meta = {}  # episode -> {enc, seed, easy_number}
with open(a.fights) as f:
    for r in csv.DictReader(f):
        meta[int(r["source_episode_id"])] = dict(enc=r["encounter"], seed=int(r["run_seed"]),
                                                  n=int(r["easy_number"]), hp=int(r["starting_hp"]))
glob = str(ROOT / "runs" / "schema=combat_v3" / "date=*" / f"id={a.prefix}-*" / "out" / "*.parquet")
rows = duckdb.sql(f"""
    select id, episode_id, any_value(won), any_value(terminal_value), any_value(starting_max_hp),
           any_value(starting_hp), any_value(final_hp)
    from read_parquet('{glob}', hive_partitioning = true, union_by_name = true)
    where row_kind = 'decision' group by id, episode_id""").fetchall()
A: dict = defaultdict(dict)  # arm -> salt -> ep -> fight
for rid, ep, won, tv, mhp, hp, fhp in rows:
    m = ID_RE.match(rid[len(a.prefix) + 1:])
    arm, salt = m["arm"], int(m["salt"] or 0)
    if arm.startswith(EXCLUDE) or ep not in meta:
        continue
    A[arm].setdefault(salt, {})[ep] = dict(won=bool(won), hpeq=float(tv) * (55 + mhp), lost=hp - fhp if won else hp)
# cost and worker identity from summary.json
COST: dict = defaultdict(dict)  # arm -> ep -> [search_seconds...]
SIMS: dict = defaultdict(dict)
SHAS = defaultdict(set)
SKIPPED = {}
for summ in (ROOT / "runs" / "schema=combat_v3").glob(f"date=*/id={a.prefix}-*/out/summary.json"):
    rid = summ.parent.parent.name[3:]
    m = ID_RE.match(rid[len(a.prefix) + 1:])
    arm = m["arm"]
    if arm.startswith(EXCLUDE):
        continue
    d = json.loads(summ.read_text())
    SHAS[d["worker"]["sha256"]].add(rid)
    SKIPPED[rid] = d["skipped_diverged"]
    for f in d.get("per_fight", []):
        if f["episode"] in meta:
            COST[arm].setdefault(f["episode"], []).append(f["search_seconds"])
            SIMS[arm].setdefault(f["episode"], []).append(f["simulations"])
# Re-timed runs (same configs and seeds, so the same play; id easy-retime-*): their search times replace the
# original run's, whose timing ran under CPU contention from another job (2026-09-29 18:31-18:46Z, exactly 2x).
RETIMED = set()
for summ in (ROOT / "runs" / "schema=combat_v3").glob("date=*/id=easy-retime-*/out/summary.json"):
    arm = ID_RE.match(summ.parent.parent.name[len("id=easy-retime-"):])["arm"]
    if arm in COST and a.prefix == "easy-cheap":
        d = json.loads(summ.read_text())
        COST[arm] = {f["episode"]: [f["search_seconds"]] for f in d["per_fight"] if f["episode"] in meta}
        RETIMED.add(arm)
# only complete runs count (a run still going has no summary.json)
for arm in list(A):
    for salt in list(A[arm]):
        if f"{a.prefix}-{arm}-s{salt}" not in SKIPPED and f"{a.prefix}-{arm}" not in SKIPPED:
            del A[arm][salt]
    if not A[arm]:
        del A[arm]
if REF not in A:
    raise SystemExit("no finished reference runs yet")


def score(arm, salts=None):
    """ep -> (mean hpeq, win prob, mean hp lost) over the arm's salts."""
    res = defaultdict(list)
    for s, fights in A[arm].items():
        if salts is None or s in salts:
            for ep, f in fights.items():
                res[ep].append(f)
    return {ep: (np.mean([f["hpeq"] for f in fs]), np.mean([f["won"] for f in fs]), np.mean([f["lost"] for f in fs]))
            for ep, fs in res.items()}


def boot(eps, values, n=10000, pooled=True):
    """(mean, lo, hi): equal-weight mean over encounters (pooled) or plain mean, cluster bootstrap over run seeds."""
    eps = list(eps)
    v = np.asarray(values, float)
    seeds = sorted({meta[e]["seed"] for e in eps})
    si = {s: i for i, s in enumerate(seeds)}
    col = np.array([si[meta[e]["seed"]] for e in eps])
    groups = [np.array([meta[e]["enc"] == enc for e in eps]) for enc in ENCS] if pooled else [np.ones(len(eps), bool)]
    groups = [g for g in groups if g.any()]

    def stat(w):  # w: (B, fights)
        return np.mean([(w[:, g] * v[g]).sum(1) / np.maximum(w[:, g].sum(1), 1e-12) for g in groups], axis=0)

    point = float(stat(np.ones((1, len(eps))))[0])
    boots = []
    for chunk in range(0, n, 2000):
        W = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=min(2000, n - chunk)).astype(np.float32)
        boots.append(stat(W[:, col]))
    lo, hi = np.quantile(np.concatenate(boots), [0.025, 0.975])
    return point, float(lo), float(hi)


def fmt(ci):
    return f"{ci[0]:+.2f} [{ci[1]:+.2f}, {ci[2]:+.2f}]"


def order_key(arm):
    m = re.match(r"(mcts|gen1|t5)-(\d+)(k?)(.*)", arm)
    if not m:
        return (9, 0, arm)
    return ({"mcts": 0, "gen1": 1, "t5": 2}[m[1]], int(m[2]) * (1000 if m[3] else 1), m[4])


ARMS = sorted(A, key=order_key)
ref = score(REF)
L: list[str] = []
out = L.append


def table(header, body):
    out("| " + " | ".join(header) + " |")
    out("|" + "|".join("---" if i == 0 else "---:" for i in range(len(header))) + "|")
    for r in body:
        out("| " + " | ".join(str(x) for x in r) + " |")
    out("")


out(f"# Easy cheap-agent analysis ({a.prefix})")
out("")
out(f"Fights: {len(meta)} listed; reference {REF} seeds {sorted(A[REF])}. Workers: "
    + "; ".join(f"sha {s[:10]} ({len(r)} runs)" for s, r in SHAS.items()) + ".")
out(f"Skipped (replay diverged) per run: {sorted(set(SKIPPED.values()))}.")
if RETIMED:
    out(f"Cost re-timed on an idle machine (original timing ran under CPU contention): {', '.join(sorted(RETIMED))}.")
out("")

# ------------------------------------------------------------------ A/A noise floor
if len(A[REF]) >= 4:
    x, y = score(REF, {0, 1}), score(REF, {2, 3})
    eps = sorted(set(x) & set(y))
    d = [y[e][0] - x[e][0] for e in eps]
    out("## A/A noise floor (MCTS 20k seeds 2-3 minus seeds 0-1)")
    out("")
    out(f"Pooled Δ HP-eq {fmt(boot(eps, d))}; SD of per-fight Δ {np.std(d):.2f} (n = {len(eps)}). "
        f"Deaths: seeds 0-1 {sum(len(eps) - sum(A[REF][s][e]['won'] for e in eps) for s in (0, 1))}, "
        f"seeds 2-3 {sum(len(eps) - sum(A[REF][s][e]['won'] for e in eps) for s in (2, 3))}.")
    out("")
    AA = d
else:
    AA = None

# ------------------------------------------------------------------ main table
summary = {}
body = []
for arm in ARMS:
    sc = score(arm)
    eps = sorted(set(sc) & set(ref))
    d = [sc[e][0] - ref[e][0] for e in eps]
    pooled = boot(eps, d)
    per_enc = {enc: boot([e for e in eps if meta[e]["enc"] == enc],
                         [x for e, x in zip(eps, d) if meta[e]["enc"] == enc], pooled=False) for enc in ENCS}
    per_n = {n: boot([e for e in eps if meta[e]["n"] == n], [x for e, x in zip(eps, d) if meta[e]["n"] == n])
             for n in (1, 2, 3)}
    plays = sum(len(f) for f in A[arm].values())
    deaths = sum(1 for f in A[arm].values() for x in f.values() if not x["won"])
    cost = [np.mean(v) for v in COST[arm].values()] if COST.get(arm) else [float("nan")]
    sims = [np.mean(v) for v in SIMS[arm].values()] if SIMS.get(arm) else [float("nan")]
    sec = float(np.mean(cost))
    summary[arm] = dict(pooled=pooled, per_enc=per_enc, per_n=per_n, deaths=deaths, plays=plays, sec=sec,
                        fps=1 / sec if sec > 0 else float("nan"), sims=float(np.mean(sims)), n=len(eps),
                        salts=sorted(A[arm]), lost=float(np.mean([sc[e][2] for e in eps])),
                        worst_enc=min(per_enc.values(), key=lambda c: c[0])[0])
    s = summary[arm]
    goal = "yes" if arm != REF and s["pooled"][0] >= -0.5 and s["worst_enc"] >= -1.0 else ("ref" if arm == REF else "no")
    body.append((arm, len(s["salts"]), s["n"], f"{s['lost']:.2f}", fmt(pooled) if arm != REF else "—",
                 *(f"{per_enc[e][0]:+.2f}" if arm != REF else "—" for e in ENCS),
                 f"{deaths}/{plays}", f"{sec * 1000:.1f}", f"{s['fps']:.0f}", f"{s['sims']:.0f}", goal))
out("## Quality vs cost, every arm (Δ = arm − MCTS 20k, HP-eq per fight; + is better)")
out("")
out("HP lost = mean over fights of (start HP − final HP), seed-averaged. ms = mean search milliseconds per fight "
    "(one thread, machine fully loaded); fights/s = 1 / that. Goal = pooled ≥ −0.5 and no encounter < −1.0.")
out("")
table(("arm", "seeds", "fights", "HP lost", "pooled Δ [95% CI]", *ENCS, "deaths/plays", "ms/fight", "fights/s",
       "sims/fight", "goal"), body)

out("## By easy-fight number (pooled Δ vs MCTS 20k; 1 = floor 1, 2-3 = later easy fights)")
out("")
table(("arm", "1st", "2nd", "3rd"), [(arm, *(fmt(summary[arm]["per_n"][n]) for n in (1, 2, 3)))
                                     for arm in ARMS if arm != REF])
out("## Per-encounter CIs")
out("")
table(("arm", *ENCS), [(arm, *(fmt(summary[arm]["per_enc"][e]) for e in ENCS)) for arm in ARMS if arm != REF])
(OUT / "analysis.md").write_text("\n".join(L) + "\n")
(OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
print("\n".join(L))

# ------------------------------------------------------------------ charts
COLORS = plt.cm.viridis
mcts = [x for x in ARMS if x.startswith("mcts-") and re.fullmatch(r"mcts-\d+k?", x)]
nets = [x for x in ARMS if x.startswith(("gen1", "t5"))]
others = [x for x in ARMS if x not in mcts and x not in nets]
show = [x for x in ("mcts-10", "mcts-50", "mcts-200", "mcts-1k", "mcts-20k", "gen1-200", "t5-50") if x in A]


def kde_line(ax, values, label, color, xs, ls="-"):
    values = np.asarray(values, float)
    if len(values) < 5 or np.std(values) == 0:
        return
    ax.plot(xs, gaussian_kde(values)(xs), label=label, color=color, ls=ls)


def color(arm):
    if arm in mcts:
        return COLORS(mcts.index(arm) / max(1, len(mcts) - 1))
    return {"gen1": "tab:red", "t5": "tab:orange"}.get(arm.split("-")[0], "tab:gray")


# 1. PDF of HP lost per fight (all seeds' plays, won fights), by encounter
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
xs = np.linspace(-5, 45, 400)
for ax, enc in zip(axes.flat, ENCS):
    for arm in show:
        v = [f["lost"] for fs in A[arm].values() for e, f in fs.items() if meta[e]["enc"] == enc and f["won"]]
        kde_line(ax, v, arm, color(arm), xs, "--" if arm in nets else "-")
    ax.set_title(enc)
    ax.set_xlabel("HP lost in the fight (net of Burning Blood's +6; won fights, every seed's play)")
    ax.tick_params(labelbottom=True)
    ax.set_ylabel("density")
axes[0, 0].legend(fontsize=8)
fig.suptitle("PDF of HP lost per fight")
fig.tight_layout()
fig.savefig(OUT / "pdf_hp_lost.png", dpi=120)
plt.close(fig)

# 2. PDF of the paired per-fight Δ vs the reference (seed-averaged), with the A/A floor; linear and log density
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
xs = np.linspace(-20, 12, 500)
for ax, log in zip(axes, (False, True)):
    for arm in show:
        if arm == REF:
            continue
        sc = score(arm)
        eps = sorted(set(sc) & set(ref))
        kde_line(ax, [sc[e][0] - ref[e][0] for e in eps], f"{arm} (mean {summary[arm]['pooled'][0]:+.2f})", color(arm),
                 xs, "--" if arm in nets else "-")
    if AA is not None:
        kde_line(ax, AA, "A/A: 20k s2-3 − s0-1", "black", xs, ":")
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("per-fight Δ HP-eq vs MCTS 20k (seed-averaged; + = better)")
    ax.set_ylabel("density")
    if log:
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 2)
        ax.set_title("same, log density (tails)")
    else:
        ax.set_title("PDF of paired per-fight difference vs the reference")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "pdf_delta.png", dpi=120)
plt.close(fig)

# 3. PDF of search time per fight (log x)
fig, ax = plt.subplots(figsize=(10, 6))
for arm in ARMS:
    if arm in COST and (arm in show or arm in nets):
        v = np.log10([max(x, 1e-6) for vs in COST[arm].values() for x in vs])
        xs = np.linspace(v.min() - 0.3, v.max() + 0.3, 300)
        kde_line(ax, v, arm, color(arm), xs, "--" if arm in nets else "-")
ax.set_xlabel("log10 search seconds per fight")
ax.set_ylabel("density")
ax.legend(fontsize=8)
ax.set_title("PDF of search time per fight (one thread, 11 workers on 6 cores / 12 threads)")
fig.tight_layout()
fig.savefig(OUT / "pdf_search_seconds.png", dpi=120)
plt.close(fig)

# 4. Frontier: quality vs throughput
fig, ax = plt.subplots(figsize=(10, 6))
for group, marker in ((mcts, "o"), (nets, "s"), (others, "^")):
    for arm in group:
        s = summary[arm]
        m, lo, hi = s["pooled"] if arm != REF else (0, 0, 0)
        ax.errorbar(s["fps"], m, yerr=[[m - lo], [hi - m]], fmt=marker, color=color(arm), capsize=3)
        ax.annotate(arm, (s["fps"], m), textcoords="offset points", xytext=(5, 4), fontsize=8)
xs_m = [summary[x]["fps"] for x in mcts]
ax.plot(xs_m, [summary[x]["pooled"][0] if x != REF else 0 for x in mcts], color="gray", lw=0.7)
ax.axhline(0, color="black", lw=0.6)
ax.axhline(-0.5, color="red", lw=0.6, ls="--", label="goal: −0.5 HP/fight")
ax.set_xscale("log")
ax.set_xlabel("fights per CPU-second (search only, log)")
ax.set_ylabel("pooled Δ HP-eq per fight vs MCTS 20k (95% CI)")
ax.legend()
ax.set_title("Cost vs quality on easy-pool fights")
fig.tight_layout()
fig.savefig(OUT / "frontier.png", dpi=120)
plt.close(fig)

# 5. MCTS budget curve per encounter
fig, ax = plt.subplots(figsize=(10, 6))
budgets = [order_key(x)[1] for x in mcts]
for i, enc in enumerate(ENCS):
    ms = [summary[x]["per_enc"][enc] if x != REF else (0, 0, 0) for x in mcts]
    c = plt.cm.tab10(i)
    ax.plot(budgets, [m[0] for m in ms], "o-", color=c, label=enc)
    ax.fill_between(budgets, [m[1] for m in ms], [m[2] for m in ms], color=c, alpha=0.12)
ax.axhline(0, color="black", lw=0.6)
ax.axhline(-0.5, color="red", lw=0.6, ls="--")
ax.set_xscale("log")
ax.set_xlabel("MCTS simulations per decision (log)")
ax.set_ylabel("Δ HP-eq per fight vs MCTS 20k (95% CI band)")
ax.legend()
ax.set_title("MCTS budget curve by encounter")
fig.tight_layout()
fig.savefig(OUT / "budget_curve.png", dpi=120)
plt.close(fig)
print(f"charts: {OUT}")
