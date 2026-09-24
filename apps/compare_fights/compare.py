#!/usr/bin/env python3
"""Paired comparison of two sets of combat_v3 fights on the candidate's fights. Spec: slop_docs/apps/compare_fights.md.

[run] baseline / candidate: queries (sts_combat_rl.query); oracle = true allows oracle rows in either.
"""
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
from apps.common.app import check_keys, flag, main, run_json, write_json
from sts_combat_rl import query

NAME = "fight_comparison_v1"
RESAMPLES, SEED = 10_000, 0
PRIMARY = "terminal_value"
TESTED = (PRIMARY, "final_hp", "potions", "turns")  # paired: diff, bootstrap CI, Wilcoxon
DESCRIBED = ("sims_per_decision", "decisions")      # compute: means only
FIGHT = ("encounter", "starting_hp", "starting_max_hp")
OUTCOME = ("won", "final_hp", "potions", "terminal_value")


def fights(sql, oracle, episodes=None):
    """episode_id -> one row per fight of the decision rows of `sql` (only `episodes` if given); the source run ids."""
    where = "row_kind = 'decision'" + (f" and episode_id in {query.sql_list(sorted(episodes))}" if episodes is not None else "")
    result, sources = {}, {}
    for row in query.rows(sql, ["episode_id", *FIGHT, *OUTCOME, "turn", "simulations_used"], where, oracle):
        episode = row["episode_id"]
        if sources.setdefault(episode, row["run_id"]) != row["run_id"]:
            raise RuntimeError(f"episode {episode}: in both {sources[episode]} and {row['run_id']}")
        fight = result.setdefault(episode, {**{c: row[c] for c in (*FIGHT, *OUTCOME)},
                                            "turns": 0, "decisions": 0, "simulations": 0})
        fight["turns"] = max(fight["turns"], row["turn"])
        fight["decisions"] += 1
        fight["simulations"] += row["simulations_used"]
    for fight in result.values():
        fight["sims_per_decision"] = fight["simulations"] / fight["decisions"]
    return result, sorted(set(sources.values()))


def paired(base, cand, rng):
    """Means, mean paired difference (candidate - baseline) with bootstrap 95% CI, Wilcoxon p."""
    diff = cand - base
    means = diff[rng.integers(0, len(diff), (RESAMPLES, len(diff)))].mean(axis=1)
    p = stats.wilcoxon(diff).pvalue if np.any(diff) else 1.0
    return {"baseline_mean": base.mean(), "candidate_mean": cand.mean(), "diff": diff.mean(),
            "ci_low": np.quantile(means, 0.025), "ci_high": np.quantile(means, 0.975), "p": p}


def inputs(config):
    """The baseline, then the candidate run ids."""
    run = config["run"]
    return [*query.run_ids(run["baseline"]), *query.run_ids(run["candidate"])]


def compare(config):
    run = config["run"]
    check_keys(run, {"id", "baseline", "candidate", "oracle"}, "run")
    oracle = flag(run, "oracle")
    cand, candidate_runs = fights(run["candidate"], oracle)
    base, baseline_runs = fights(run["baseline"], oracle, cand.keys())
    if missing := sorted(cand.keys() - base.keys()):
        sys.exit(f"{len(missing)} candidate fights missing from the baseline, e.g. {missing[:5]}")
    episodes = sorted(cand)
    if bad := [e for e in episodes if any(base[e][c] != cand[e][c] for c in FIGHT)]:
        sys.exit(f"{len(bad)} fights start differently in the two runs (not the same fight), e.g. {bad[:5]}")
    column = lambda side, metric: np.array([float(side[e][metric]) for e in episodes])
    rng = np.random.default_rng(SEED)
    metrics = {m: paired(column(base, m), column(cand, m), rng) for m in TESTED}
    for m in DESCRIBED:
        metrics[m] = {"baseline_mean": column(base, m).mean(), "candidate_mean": column(cand, m).mean()}
    only_base = sum(base[e]["won"] and not cand[e]["won"] for e in episodes)
    only_cand = sum(cand[e]["won"] and not base[e]["won"] for e in episodes)
    wins = {"baseline_mean": column(base, "won").mean(), "candidate_mean": column(cand, "won").mean(),
            "only_baseline": only_base, "only_candidate": only_cand,
            "p": stats.binomtest(only_cand, only_base + only_cand).pvalue if only_base + only_cand else 1.0}
    diff = column(cand, PRIMARY) - column(base, PRIMARY)
    summary = {"schema": NAME, "config": config, "baseline_runs": baseline_runs, "candidate_runs": candidate_runs,
               "n": len(episodes), "won": wins, **metrics,
               PRIMARY + "_better_worse_equal": [int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())]}
    pairs = pa.Table.from_pylist([
        {"episode_id": e, "encounter": cand[e]["encounter"],
         **{f"{side}_{m}": s[e][m] for m in ("won", *TESTED, *DESCRIBED) for side, s in (("baseline", base), ("candidate", cand))}}
        for e in episodes])
    return summary, pairs, base, cand


def report(summary, base, cand):
    config, n = summary["config"]["run"], summary["n"]
    runs = {k: summary[f"{k}_runs"] for k in ("baseline", "candidate")}
    inputs = {k: [run_json(run_id)["inputs"] for run_id in runs[k]] for k in runs}
    t = summary[PRIMARY]
    better, worse, equal = summary[PRIMARY + "_better_worse_equal"]
    w = summary["won"]
    lines = [
        f"# {', '.join(runs['candidate'])} vs {', '.join(runs['baseline'])}", "",
        f"- baseline:  `{config['baseline']}` → `{', '.join(runs['baseline'])}` (inputs: {inputs['baseline'] or 'none'})",
        f"- candidate: `{config['candidate']}` → `{', '.join(runs['candidate'])}` (inputs: {inputs['candidate'] or 'none'})",
        f"- fights compared: **{n}** (the candidate's fights)", "",
        f"**{PRIMARY}: candidate − baseline = {t['diff']:+.3f} (95% CI {t['ci_low']:+.3f} to {t['ci_high']:+.3f}), "
        f"Wilcoxon p = {t['p']:.3g}, n = {n}**; better / worse / equal: {better} / {worse} / {equal}", "",
        "| metric | baseline | candidate | diff | 95% CI | p | |", "|---|---|---|---|---|---|---|",
        f"| won | {w['baseline_mean']:.1%} | {w['candidate_mean']:.1%} | only baseline {w['only_baseline']}, "
        f"only candidate {w['only_candidate']} | | {w['p']:.3g} | exact McNemar |",
    ]
    for m in TESTED:
        s = summary[m]
        lines.append(f"| {m} | {s['baseline_mean']:.3f} | {s['candidate_mean']:.3f} | {s['diff']:+.3f} | "
                     f"{s['ci_low']:+.3f} to {s['ci_high']:+.3f} | {s['p']:.3g} | {'primary' if m == PRIMARY else 'descriptive'} |")
    for m in DESCRIBED:
        s = summary[m]
        lines.append(f"| {m} | {s['baseline_mean']:,.1f} | {s['candidate_mean']:,.1f} | | | | compute |")
    worst = sorted(cand, key=lambda e: cand[e][PRIMARY] - base[e][PRIMARY])[:5]
    lines += ["", "## Worst regressions", "", "| episode_id | encounter | baseline tv | candidate tv | baseline hp | candidate hp |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {e} | {cand[e]['encounter']} | {base[e][PRIMARY]:.3f} | {cand[e][PRIMARY]:.3f} | "
              f"{base[e]['final_hp']} | {cand[e]['final_hp']} |" for e in worst]
    return "\n".join(lines) + "\n"


def run(config, config_path, out):
    summary, pairs, base, cand = compare(config)
    text = report(summary, base, cand)
    (out / "report.md").write_text(text)
    write_json(out / "summary.json", summary)
    pq.write_table(pairs, out / "pairs.parquet")
    print(text, flush=True)


if __name__ == "__main__":
    main(run, inputs)
