#!/usr/bin/env python3
"""Hard-budget study: tables and figures from the finished runs.

    PYTHONPATH=python .venv/bin/python experiments/act-1-combat/2026-09-29-hard-budget/report.py

HP lost = starting HP - final HP, a death counting as all starting HP (after Burning Blood's +6 on a win, so it can be
negative). Per fight averaged over the arm's search seeds. Pooled = encounters weighted by how often they occur in
real runs (hard-fight counts in act1-eval-mcts-a20). Per run = pooled x 1.6 hard fights. Δ = paired per-fight
difference (+ = loses more HP). 95% CIs: cluster bootstrap over source run seeds (10,000 resamples).
Search ms = search_seconds per fight (move choosing only; one thread; 11 workers on a 6-core/12-thread Ryzen 3600X).
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import duckdb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
RES = HERE / "results"
PREFIX = "hard-budget"
FREQ = {"looter": 222, "blue_slaver": 209, "two_fungi_beasts": 204, "exordium_thugs": 183, "large_slime": 176,
        "three_louse": 164, "exordium_wildlife": 156, "lots_of_slimes": 113, "gremlin_gang": 105, "red_slaver": 104}
ENCS = tuple(FREQ)
NAMES = {e: e.replace("_", " ").title() for e in ENCS}
W = np.array([FREQ[e] for e in ENCS], float) / sum(FREQ.values())
PER_RUN = 1.6
rng = np.random.default_rng(2026)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# ------------------------------------------------------------------ load
meta = {int(r["source_episode_id"]): (r["encounter"], int(r["run_seed"])) for r in csv.DictReader(open(HERE / "fights.csv"))}
done, cost = set(), defaultdict(list)
for s in (ROOT / "runs/schema=combat_v3").glob(f"date=*/id={PREFIX}-mcts-*/out/summary.json"):
    rid = s.parent.parent.name[3:]
    done.add(rid)
    cost[re.fullmatch(rf"{PREFIX}-(.+)-s\d+", rid)[1]] += [f["search_seconds"] for f in json.loads(s.read_text())["per_fight"]]
rows = duckdb.sql(f"""
    select id, episode_id, any_value(starting_hp), any_value(final_hp), any_value(won)
    from read_parquet('{ROOT}/runs/schema=combat_v3/date=*/id={PREFIX}-mcts-*/out/*.parquet', hive_partitioning = true)
    where row_kind = 'decision' and decision_index = 0 group by id, episode_id""").fetchall()
plays = defaultdict(lambda: defaultdict(dict))  # arm -> ep -> seed -> (hp lost, died)
for rid, ep, hp, fhp, won in rows:
    if rid in done and ep in meta:
        m = re.fullmatch(rf"{PREFIX}-(.+)-s(\d+)", rid)
        plays[m[1]][ep][int(m[2])] = (hp - fhp if won else hp, 0 if won else 1)
sims = lambda a: int(a[5:-1]) * 1000 if a.endswith("k") else int(a[5:])
ARMS = sorted(plays, key=sims)
TOP = ARMS[-1]
common = set.intersection(*(set(plays[a]) for a in ARMS))  # fights every arm played (same skipped set)


def per_fight(arm, i):  # i = 0: hp lost, 1: died; seed-averaged
    return {e: np.mean([v[i] for v in plays[arm][e].values()]) for e in common}


def boot(values):
    """Frequency-weighted pooled mean and 95% CI; cluster bootstrap over run seeds."""
    eps = sorted(values)
    v = np.array([values[e] for e in eps])
    seeds = sorted({meta[e][1] for e in eps})
    col = np.searchsorted(seeds, [meta[e][1] for e in eps])
    groups = [(w, np.array([meta[e][0] == enc for e in eps])) for w, enc in zip(W, ENCS)]
    Wb = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=10000).astype(np.float32)[:, col]
    stat = lambda w: sum(wt * (w[:, g] * v[g]).sum(1) / np.maximum(w[:, g].sum(1), 1e-9) for wt, g in groups)
    lo, hi = np.quantile(stat(Wb), [0.025, 0.975])
    return float(stat(np.ones((1, len(eps))))[0]), float(lo), float(hi)


def enc_mean(values, enc):
    return float(np.mean([v for e, v in values.items() if meta[e][0] == enc]))


f = lambda c, s="+": f"{c[0]:{s}.2f} [{c[1]:{s}.2f}, {c[2]:{s}.2f}]"
R = {}
for a in ARMS:
    hp, died = per_fight(a, 0), per_fight(a, 1)
    R[a] = dict(hp=hp, died=died, HP=boot(hp), D=boot(died), ms=1000 * float(np.mean(cost[a])),
                deaths=sum(v[1] for e in common for v in plays[a][e].values()),
                n_plays=sum(len(plays[a][e]) for e in common), seeds=max(len(plays[a][e]) for e in common))
for i, a in enumerate(ARMS):
    R[a]["vs_top"] = None if a == TOP else boot({e: R[a]["hp"][e] - R[TOP]["hp"][e] for e in common})
    up = ARMS[i + 1] if i + 1 < len(ARMS) else None
    R[a]["vs_up"] = None if up is None else (up, boot({e: R[a]["hp"][e] - R[up]["hp"][e] for e in common}))
aa = None
if all(len(plays[TOP][e]) >= 2 for e in common):
    aa = {e: plays[TOP][e][1][0] - plays[TOP][e][0][0] for e in common}

L = [f"# Hard-pool fights: HP lost by MCTS budget (n = {len(common)} fights, 10 encounters)", "",
     "HP lost counts a death as all starting HP. Pooled = encounters weighted by real-run frequency. "
     "Δ = paired per-fight difference (+ = loses more HP). [95% CI, cluster bootstrap over run seeds]. "
     f"Per run = × {PER_RUN} hard fights.", "",
     f"| MCTS sims | seeds | HP lost / fight | deaths | Δ vs {TOP[5:]} | Δ vs next budget up | per run vs {TOP[5:]} | search ms / fight |",
     "|---:|---:|---:|---:|---:|---:|---:|---:|"]
for a in ARMS:
    r = R[a]
    vt = "—" if r["vs_top"] is None else f(r["vs_top"])
    vu = "—" if r["vs_up"] is None else f"{f(r['vs_up'][1])} (vs {r['vs_up'][0][5:]})"
    pr = "—" if r["vs_top"] is None else f"{r['vs_top'][0] * PER_RUN:+.2f}"
    L.append(f"| {a[5:]} | {r['seeds']} | {f(r['HP'], '')} | {r['deaths']}/{r['n_plays']} ({100 * r['D'][0]:.1f}%) "
             f"| {vt} | {vu} | {pr} | {r['ms']:.0f} |")
if aa:
    c = boot(aa)
    L += ["", f"Noise check: {TOP[5:]} seed 1 minus seed 0 (single plays, same fights) = {f(c)} HP/fight; "
              f"per-fight SD {np.std(list(aa.values())):.1f} HP."]
L += ["", "## HP lost per fight by encounter (seed-averaged; deaths in brackets)", "",
      "| encounter | share | " + " | ".join(a[5:] for a in ARMS) + " |", "|---|---:|" + "---:|" * len(ARMS)]
for enc, w in zip(ENCS, W):
    cells = []
    for a in ARMS:
        d = sum(v[1] for e in common if meta[e][0] == enc for v in plays[a][e].values())
        cells.append(f"{enc_mean(R[a]['hp'], enc):.1f}" + (f" ({d})" if d else ""))
    L.append(f"| {NAMES[enc]} | {100 * w:.0f}% | " + " | ".join(cells) + " |")
text = "\n".join(L) + "\n"
(RES / "tables.md").write_text(text)
print(text)

# ------------------------------------------------------------------ figures
B = [sims(a) for a in ARMS]
ticks = [a[5:] for a in ARMS]

# Fig 1: HP lost and death rate vs budget
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
m = [R[a]["HP"] for a in ARMS]
dv = [R[a]["vs_top"] or (0.0, 0.0, 0.0) for a in ARMS]
ax.errorbar(B, [x[0] for x in dv], yerr=[[x[0] - x[1] for x in dv], [x[2] - x[0] for x in dv]], fmt="o-", capsize=3)
for b, x, a in zip(B, dv, ARMS):
    ax.annotate(f"{x[0]:+.2f}\n({R[a]['HP'][0]:.2f} lost)" if a != TOP else f"ref\n({R[a]['HP'][0]:.2f} lost)", (b, x[0]),
                textcoords="offset points", xytext=(7, 4), fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_xscale("log")
ax.set_xticks(B, ticks)
ax.minorticks_off()
ax.set_xlabel("MCTS simulations per decision")
ax.set_ylabel(f"extra HP lost per fight vs MCTS {TOP[5:]} (paired)")
ax.set_title(f"Extra HP lost vs {TOP[5:]}, same fights (95% CI)")
d = [R[a]["D"] for a in ARMS]
ax2.errorbar(B, [100 * x[0] for x in d], yerr=[[100 * (x[0] - x[1]) for x in d], [100 * (x[2] - x[0]) for x in d]],
             fmt="o-", capsize=3, color="tab:red")
for b, a, x in zip(B, ARMS, d):
    ax2.annotate(f"{100 * x[0]:.1f}% ({R[a]['deaths']})", (b, 100 * x[0]), textcoords="offset points", xytext=(7, 5), fontsize=9)
ax2.set_xscale("log")
ax2.set_xticks(B, ticks)
ax2.minorticks_off()
ax2.set_ylim(0, None)
ax2.set_xlabel("MCTS simulations per decision")
ax2.set_ylabel("death rate, % of fights")
ax2.set_title("Deaths (95% CI; number of deaths in brackets)")
fig.suptitle(f"Fig 1. Hard-pool fights by MCTS budget (n = {len(common)} fights × 2 seeds; encounters weighted by frequency)")
fig.tight_layout()
fig.savefig(RES / "fig1_hp_and_deaths_vs_budget.png", dpi=130)
plt.close(fig)

# Fig 2: per encounter
fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharex=True)
for ax, enc in zip(axes.flat, sorted(ENCS, key=lambda e: -enc_mean(R[ARMS[0]]["hp"], e))):
    ys = [enc_mean(R[a]["hp"], enc) for a in ARMS]
    ax.plot(B, ys, "o-")
    dth = [sum(v[1] for e in common if meta[e][0] == enc for v in plays[a][e].values()) for a in ARMS]
    for b, y, k in zip(B, ys, dth):
        if k:
            ax.annotate(f"†{k}", (b, y), textcoords="offset points", xytext=(0, 6), fontsize=8, color="tab:red", ha="center")
    ax.set_title(f"{NAMES[enc]} ({100 * FREQ[enc] / sum(FREQ.values()):.0f}%)", fontsize=10)
    ax.set_xscale("log")
    ax.set_xticks(B, ticks, fontsize=8)
    ax.minorticks_off()
    ax.set_ylim(0, None)
for ax in axes[:, 0]:
    ax.set_ylabel("HP lost per fight")
fig.suptitle("Fig 2. HP lost per fight by encounter (≈100 fights each, seed-averaged; †n = deaths over 2 seeds; "
             "share of hard fights in brackets)")
fig.tight_layout()
fig.savefig(RES / "fig2_by_encounter.png", dpi=130)
plt.close(fig)

# Fig 3: cost vs HP
fig, ax = plt.subplots(figsize=(9, 5.5))
xs = [R[a]["ms"] for a in ARMS]
ax.errorbar(xs, [x[0] for x in m], yerr=[[x[0] - x[1] for x in m], [x[2] - x[0] for x in m]], fmt="o-", capsize=3)
for x, a, c in zip(xs, ARMS, m):
    ax.annotate(f"MCTS {a[5:]}", (x, c[0]), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("search time per fight, ms (one CPU thread, log scale)")
ax.set_ylabel("HP lost per fight")
top = ax.secondary_xaxis("top", functions=(lambda v: 1000 / np.maximum(v, 1e-9), lambda v: 1000 / np.maximum(v, 1e-9)))
top.set_xlabel("fights per CPU-second")
ax.set_title("Fig 3. Cost vs HP lost on hard-pool fights", pad=10)
fig.tight_layout()
fig.savefig(RES / "fig3_cost_vs_hp.png", dpi=130)
plt.close(fig)

# Fig 4: one play vs the top budget on the same fight (seed 0 vs seed 0; noise = top seed 1 vs seed 0)
fig, ax = plt.subplots(figsize=(9, 5.5))
ref0 = {e: plays[TOP][e][0][0] for e in common}
series = ([(f"{TOP[5:]}, other seed\n(noise)", {e: plays[TOP][e][1][0] for e in common})] if aa else []) + \
         [(a[5:], {e: plays[a][e][0][0] for e in common}) for a in ARMS if a != TOP]
for x, (name, v) in enumerate(series):
    diff = np.array([v[e] - ref0[e] for e in common])
    worse, same, better = (diff > 2).mean(), (abs(diff) <= 2).mean(), (diff < -2).mean()
    ax.bar(x, worse, color="tab:red", label="loses > 2 HP more" if x == 0 else None)
    ax.bar(x, same, bottom=worse, color="lightgray", label="within 2 HP" if x == 0 else None)
    ax.bar(x, better, bottom=worse + same, color="tab:green", label="loses > 2 HP less" if x == 0 else None)
    ax.text(x, worse / 2, f"{worse:.0%}", ha="center", va="center", fontsize=9, color="white")
    ax.text(x, worse + same + better / 2, f"{better:.0%}", ha="center", va="center", fontsize=9, color="white")
ax.set_xticks(range(len(series)), [s[0] for s in series])
ax.set_xlabel("MCTS simulations per decision (one play per fight, seed 0)")
ax.set_ylabel("share of fights")
ax.set_ylim(0, 1)
ax.set_title(f"Fig 4. One play vs MCTS {TOP[5:]} on the same fight: how often is it clearly worse or better?")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3)
fig.tight_layout()
fig.savefig(RES / "fig4_worse_or_better.png", dpi=130)
plt.close(fig)
print("figures:", ", ".join(sorted(p.name for p in RES.glob("fig*.png"))))
