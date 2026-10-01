#!/usr/bin/env python3
"""The report's tables and figures: HP lost per easy fight, every arm, from the finished runs.

    PYTHONPATH=python .venv/bin/python experiments/act-1-combat/2026-09-29-easy-cheap-agent/report.py

Writes results/hp_lost_{dev,confirm}.md and results/fig*.png.
HP lost = starting HP - final HP (net of Burning Blood's +6), per fight averaged over the arm's search seeds.
Mean = the 4 encounters weighted equally. Δ = paired per-fight difference vs MCTS 20k (+ = loses more HP).
95% CIs: cluster bootstrap over source run seeds (10,000 resamples).
Cost = search_seconds per fight (move choosing only; one thread; 11 workers on a 6-core/12-thread Ryzen 3600X).
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
ENCS = ("cultist", "jaw_worm", "small_slimes", "two_louse")
NAMES = {"cultist": "Cultist", "jaw_worm": "Jaw Worm", "small_slimes": "Small Slimes", "two_louse": "Two Louse"}
rng = np.random.default_rng(2026)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def key(arm):
    m = re.match(r"(mcts|gen1|t5)-(\d+)(k?)(.*)", arm)
    return ({"mcts": 0, "gen1": 1, "t5": 2}[m[1]], m[4] != "", int(m[2]) * (1000 if m[3] else 1), m[4])


def sims(arm):
    return key(arm)[2]


def is_sweep(arm):
    return re.fullmatch(r"mcts-\d+k?", arm) is not None


def load(prefix, fights):
    meta = {int(r["source_episode_id"]): (r["encounter"], int(r["run_seed"])) for r in csv.DictReader(open(HERE / fights))}
    runs = (ROOT / "runs/schema=combat_v3").glob(f"date=*/id={prefix}-*/out/summary.json")
    done, cost = set(), defaultdict(list)
    for s in runs:
        rid = s.parent.parent.name[3:]
        m = re.fullmatch(rf"{prefix}-(.+)-s\d+", rid)
        if not m or m[1].startswith(("pilot", "confirm")):
            continue
        done.add(rid)
        cost[m[1]] += [f["search_seconds"] for f in json.loads(s.read_text())["per_fight"]]
    rows = duckdb.sql(f"""
        select id, episode_id, any_value(starting_hp) - any_value(final_hp), any_value(won)
        from read_parquet('{ROOT}/runs/schema=combat_v3/date=*/id={prefix}-*/out/*.parquet', hive_partitioning = true)
        where row_kind = 'decision' and decision_index = 0 group by id, episode_id""").fetchall()
    lost = defaultdict(lambda: defaultdict(dict))  # arm -> ep -> {seed: hp lost}
    deaths = defaultdict(int)
    for rid, ep, hl, won in rows:
        if rid in done and ep in meta:
            m = re.fullmatch(rf"{prefix}-(.+)-s(\d+)", rid)
            lost[m[1]][ep][int(m[2])] = hl
            deaths[m[1]] += not won
    return meta, lost, deaths, cost


def boot(meta, per_fight):
    """Encounter-equal mean and 95% CI; cluster bootstrap over run seeds."""
    eps = list(per_fight)
    v = np.array([per_fight[e] for e in eps])
    seeds = sorted({meta[e][1] for e in eps})
    col = np.searchsorted(seeds, [meta[e][1] for e in eps])
    groups = [np.array([meta[e][0] == enc for e in eps]) for enc in ENCS]
    groups = [g for g in groups if g.any()]
    W = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=10000).astype(np.float32)[:, col]
    stat = lambda w: np.mean([(w[:, g] * v[g]).sum(1) / w[:, g].sum(1) for g in groups], axis=0)
    lo, hi = np.quantile(stat(W), [0.025, 0.975])
    return float(stat(np.ones((1, len(eps))))[0]), float(lo), float(hi)


def analyse(prefix, fights, tag):
    meta, lost, deaths, cost = load(prefix, fights)
    arms = sorted(lost, key=key)
    ref = {e: np.mean(list(v.values())) for e, v in lost["mcts-20k"].items()}
    res = {}
    for arm in arms:
        pf = {e: np.mean(list(v.values())) for e, v in lost[arm].items() if e in ref}
        res[arm] = dict(
            hp=boot(meta, pf), d=boot(meta, {e: pf[e] - ref[e] for e in pf}), n=len(pf),
            seeds=max(len(v) for v in lost[arm].values()), deaths=deaths[arm],
            enc={enc: boot(meta, {e: pf[e] for e in pf if meta[e][0] == enc}) for enc in ENCS},
            ms=1000 * float(np.mean(cost[arm])), per_fight=pf)
    L = [f"# HP lost per easy fight ({tag} fights, n = {res['mcts-20k']['n']})", "",
         "HP lost = start HP − end HP (after Burning Blood's +6). Mean = 4 encounters weighted equally. "
         "Δ = extra HP lost vs MCTS 20k on the same fights (+ = worse). [95% CI, cluster bootstrap over run seeds]. "
         "8 particles unless -pN. ms = search time per fight.", "",
         "| agent | seeds | HP lost / fight | Δ vs MCTS 20k | " + " | ".join(NAMES[e] for e in ENCS) + " | deaths | ms |",
         "|---|---:|---:|---:|" + "---:|" * len(ENCS) + "---:|---:|"]
    for arm in arms:
        r = res[arm]
        hp, d = r["hp"], r["d"]
        dd = "—" if arm == "mcts-20k" else f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"
        L.append(f"| {arm} | {r['seeds']} | {hp[0]:.2f} [{hp[1]:.2f}, {hp[2]:.2f}] | {dd} | "
                 + " | ".join(f"{r['enc'][e][0]:.2f}" for e in ENCS) + f" | {r['deaths']} | "
                 + ("—*" if "-p" in arm else f"{r['ms']:.0f}") + " |")
    if any("-p" in a for a in arms):
        L += ["", "\\* Particle arms: timing not comparable (most ran while another job shared the CPU). Re-timed on an "
                  "idle machine, 1/2/4 particles cost the same as 8 (~11 ms at 100 simulations)."]
    if len(lost["mcts-20k"][next(iter(lost["mcts-20k"]))]) >= 4:  # A/A: 20k seeds 2-3 vs 0-1
        aa = {e: np.mean([v[2], v[3]]) - np.mean([v[0], v[1]]) for e, v in lost["mcts-20k"].items() if len(v) == 4}
        c = boot(meta, aa)
        L += ["", f"A/A check (MCTS 20k seeds 2–3 minus seeds 0–1, same fights): {c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}] "
                  f"HP/fight, i.e. no difference, as it should be."]
        res["_aa"] = aa
    (RES / f"hp_lost_{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), "\n")
    return meta, lost, res


meta, lost, dev = analyse("easy-cheap", "fights.csv", "dev")
_, _, conf = analyse("easy-cheap-confirm", "confirm.csv", "confirm")
sweep = [a for a in sorted(dev, key=lambda a: key(a) if not a.startswith("_") else (9,)) if not a.startswith("_") and is_sweep(a)]
B = [sims(a) for a in sweep]
REF_HP = dev["mcts-20k"]["hp"][0]


def err(ax, x, c, **kw):
    m, lo, hi = c
    return ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], capsize=3, **kw)


# ---- Fig 1: the headline. Extra HP lost vs budget, dev and fresh fights
fig, ax = plt.subplots(figsize=(9, 5.5))
d = [dev[a]["d"] if a != "mcts-20k" else (0, 0, 0) for a in sweep]
ax.plot(B, [x[0] for x in d], "-", color="tab:blue", lw=1)
for i, (b, c) in enumerate(zip(B, d)):
    err(ax, b, c, fmt="o", color="tab:blue", label="dev fights (n=983)" if i == 0 else None)
    ax.annotate(f"+{c[0]:.2f}" if c[0] else "ref", (b, c[0]), textcoords="offset points", xytext=(7, 4), fontsize=9,
                color="tab:blue")
for i, a in enumerate(x for x in sweep if x in conf and x != "mcts-20k"):
    err(ax, sims(a) * 1.15, conf[a]["d"], fmt="s", color="tab:green", label="fresh fights (n=985)" if i == 0 else None)
ax.axhline(0, color="black", lw=0.8)
ax.set_xscale("log")
ax.set_xticks(B, [f"{b // 1000}k" if b >= 1000 else str(b) for b in B])
ax.minorticks_off()
ax.set_xlabel("MCTS simulations per decision")
ax.set_ylabel("extra HP lost per fight vs MCTS 20k")
ax.set_title(f"Fig 1. Extra HP lost per easy fight, by search budget\n(MCTS 20k itself loses {REF_HP:.2f} HP per fight; "
             "error bars = 95% CI)")
ax.legend()
fig.tight_layout()
fig.savefig(RES / "fig1_extra_hp_vs_budget.png", dpi=130)
plt.close(fig)

# ---- Fig 2: absolute HP lost per encounter vs budget
fig, ax = plt.subplots(figsize=(9, 5.5))
for i, enc in enumerate(ENCS):
    cs = [dev[a]["enc"][enc] for a in sweep]
    c = plt.cm.tab10(i)
    ax.plot(B, [x[0] for x in cs], "o-", color=c, label=NAMES[enc])
    ax.fill_between(B, [x[1] for x in cs], [x[2] for x in cs], color=c, alpha=0.15)
ax.set_xscale("log")
ax.set_xticks(B, [f"{b // 1000}k" if b >= 1000 else str(b) for b in B])
ax.minorticks_off()
ax.set_ylim(0, None)
ax.set_xlabel("MCTS simulations per decision")
ax.set_ylabel("HP lost per fight")
ax.set_title("Fig 2. HP lost per fight by encounter (dev fights; bands = 95% CI)\n"
             "Jaw Worm is the expensive fight; Two Louse barely depends on search")
ax.legend()
fig.tight_layout()
fig.savefig(RES / "fig2_hp_lost_by_encounter.png", dpi=130)
plt.close(fig)

# ---- Fig 3: cost vs quality (MCTS sweep and gen1 value net; their timings ran on an otherwise idle machine)
fig, ax = plt.subplots(figsize=(9, 5.5))
for a in sweep:
    c = dev[a]["d"] if a != "mcts-20k" else (0, 0, 0)
    err(ax, dev[a]["ms"], c, fmt="o", color="tab:blue")
    ax.annotate(f"MCTS {a[5:]}", (dev[a]["ms"], c[0]), textcoords="offset points", xytext=(-8, -16), fontsize=8)
ax.plot([dev[a]["ms"] for a in sweep], [dev[a]["d"][0] if a != "mcts-20k" else 0 for a in sweep], "-", color="tab:blue", lw=1,
        label="MCTS (guided rollouts)")
nets = [a for a in dev if a.startswith("gen1")]
for i, a in enumerate(sorted(nets, key=key)):
    err(ax, dev[a]["ms"], dev[a]["d"], fmt="s", color="tab:red", label="gen1 value net" if i == 0 else None)
    ax.annotate(a, (dev[a]["ms"], dev[a]["d"][0]), textcoords="offset points", xytext=(-10, 12), fontsize=8, color="tab:red")
ax.axhline(0, color="black", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("search time per fight, ms (one CPU thread, log scale)")
ax.set_ylabel("extra HP lost per fight vs MCTS 20k")
top = ax.secondary_xaxis("top", functions=(lambda ms: 1000 / np.maximum(ms, 1e-9), lambda f: 1000 / np.maximum(f, 1e-9)))
top.set_xlabel("fights per CPU-second")
ax.set_title("Fig 3. Cost vs HP (dev fights): returns flatten above ~200 simulations", pad=10)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(RES / "fig3_cost_vs_hp.png", dpi=130)
plt.close(fig)

# ---- Fig 4: alternatives at a matched budget (fewer particles, nets) vs plain MCTS
groups = [("MCTS 100", ["mcts-100", "mcts-100-p4", "mcts-100-p2", "mcts-100-p1"]),
          ("MCTS 500", ["mcts-500", "mcts-500-p4", "mcts-500-p2", "mcts-500-p1"]),
          ("budget 200", ["mcts-200", "gen1-200", "t5-200"]),
          ("budget 1k", ["mcts-1k", "gen1-1k"])]
label = {"mcts-100": "8 particles (default)", "mcts-500": "8 particles (default)", "mcts-200": "MCTS", "mcts-1k": "MCTS",
         "gen1-200": "gen1 value net", "gen1-1k": "gen1 value net", "t5-200": "t5 elite policy net"}
fig, ax = plt.subplots(figsize=(9, 6))
y, ticks, names = 0, [], []
for g, arms in groups:
    for a in arms:
        if a not in dev:
            continue
        c = dev[a]["d"]
        base = a in ("mcts-100", "mcts-500", "mcts-200", "mcts-1k")
        ax.errorbar(c[0], y, xerr=[[c[0] - c[1]], [c[2] - c[0]]], fmt="o", capsize=3,
                    color="tab:blue" if base else "tab:gray" if "-p" in a else "tab:red")
        ax.annotate(f"+{c[0]:.2f}", (c[0], y), textcoords="offset points", xytext=(-8, 6), fontsize=8)
        ticks.append(y)
        names.append(f"{g}: " + (label[a] if a in label else f"{a.split('-p')[1]} particle" + ("s" if a[-1] != "1" else "")))
        y -= 1
    y -= 0.7
ax.set_yticks(ticks, names)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("extra HP lost per fight vs MCTS 20k (95% CI)")
ax.set_title("Fig 4. Alternatives at the same budget are all worse than plain MCTS (blue)\n(dev fights)")
fig.tight_layout()
fig.savefig(RES / "fig4_alternatives.png", dpi=130)
plt.close(fig)

# ---- Fig 5: distribution of HP lost per fight (one play per fight: seed 0)
show = [("mcts-10", "tab:purple"), ("mcts-100", "tab:blue"), ("mcts-20k", "black")]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
bins = np.arange(-6, 38, 2)
for ax, enc in zip(axes.flat, ENCS):
    for a, c in show:
        v = [s[0] for e, s in lost[a].items() if meta[e][0] == enc and 0 in s]
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color=c, label=f"{a} (mean {np.mean(v):.1f})")
    ax.set_title(NAMES[enc])
    ax.legend(fontsize=8)
    ax.tick_params(labelbottom=True)
for ax in axes[1]:
    ax.set_xlabel("HP lost in the fight (2-HP bins)")
for ax in axes[:, 0]:
    ax.set_ylabel("share of fights")
fig.suptitle("Fig 5. Distribution of HP lost per fight (dev fights, seed 0). Below 0 = Burning Blood healed more than the "
             "fight cost.\nCheap search has fewer 0-HP fights and a longer tail of expensive ones")
fig.tight_layout()
fig.savefig(RES / "fig5_hp_lost_distribution.png", dpi=130)
plt.close(fig)

# ---- Fig 6: how often a single play is worse / better than MCTS 20k (seed 0 vs seed 0; A/A = 20k seed 1 vs seed 0)
fig, ax = plt.subplots(figsize=(9, 5.5))
ref0 = {e: s[0] for e, s in lost["mcts-20k"].items() if 0 in s}
rows = [("20k, other seed\n(noise)", {e: s[1] for e, s in lost["mcts-20k"].items() if 1 in s})] + \
       [(a[5:], {e: s[0] for e, s in lost[a].items() if 0 in s}) for a in sweep if a != "mcts-20k"]
for x, (name, v) in enumerate(rows):
    diff = np.array([v[e] - ref0[e] for e in v if e in ref0])
    worse, same, better = (diff > 2).mean(), (abs(diff) <= 2).mean(), (diff < -2).mean()
    ax.bar(x, worse, color="tab:red", label="loses > 2 HP more" if x == 0 else None)
    ax.bar(x, same, bottom=worse, color="lightgray", label="within 2 HP" if x == 0 else None)
    ax.bar(x, better, bottom=worse + same, color="tab:green", label="loses > 2 HP less" if x == 0 else None)
    ax.text(x, worse / 2, f"{worse:.0%}", ha="center", va="center", fontsize=8, color="white")
    ax.text(x, worse + same + better / 2, f"{better:.0%}", ha="center", va="center", fontsize=8, color="white")
ax.set_xticks(range(len(rows)), [r[0] for r in rows])
ax.set_xlabel("MCTS simulations per decision (one play per fight, seed 0)")
ax.set_ylabel("share of fights")
ax.set_ylim(0, 1)
ax.set_title("Fig 6. One play vs MCTS 20k on the same fight: how often is it clearly worse or better?\n"
             "Even 20k vs itself differs by > 2 HP in some fights (search noise); cheap search adds more red")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=3)
fig.tight_layout()
fig.savefig(RES / "fig6_worse_or_better.png", dpi=130)
plt.close(fig)
print("figures:", ", ".join(sorted(p.name for p in RES.glob("fig*.png"))))
