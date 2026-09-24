#!/usr/bin/env python3
"""Paired comparison of two combat_v3 runs on the candidate's fights. Spec: slop_docs/apps/compare_fights.md."""
import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy import stats
from sts_combat_rl.run import run_dir, run_parquet
from sts_combat_rl.schemas.combat_v3 import NAME as COMBAT_V3

NAME = "fight_comparison_v1"
RESAMPLES, SEED = 10_000, 0
PRIMARY = "terminal_value"
TESTED = (PRIMARY, "final_hp", "potions", "turns")  # paired: diff, bootstrap CI, Wilcoxon
DESCRIBED = ("sims_per_decision", "decisions")      # compute: means only
FIGHT = ("encounter", "starting_hp", "starting_max_hp")


def run_ids(value, side):
    """One or more combat_v3 run ids from a comparison config side."""
    ids = [value] if isinstance(value, str) else value
    if not isinstance(ids, list) or not ids or not all(isinstance(run_id, str) for run_id in ids):
        sys.exit(f"{side}: needs one or more combat_v3 run ids")
    return ids


def fights(run_ids, episodes=None):
    """episode_id -> one row per fight, merged from `run_ids` (only `episodes` if given)."""
    result, sources = {}, {}
    for run_id in run_ids:
        record = json.loads((run_dir(run_id) / "run.json").read_text())
        if record["schema"] != COMBAT_V3:
            sys.exit(f"{run_id}: schema {record['schema']}, need {COMBAT_V3}")
        where = ds.field("row_kind") == "decision"
        if episodes is not None:
            where &= ds.field("episode_id").isin(sorted(episodes))
        rows = ds.dataset(run_parquet(run_id), format="parquet").to_table(
            columns=["episode_id", *FIGHT, "won", "final_hp", "potions", "terminal_value", "turn", "simulations_used"],
            filter=where).to_pylist()
        for row in rows:
            episode = row["episode_id"]
            if episode in sources and sources[episode] != run_id:
                raise RuntimeError(f"episode {episode}: present in more than one source run")
            sources[episode] = run_id
            fight = result.setdefault(episode, {
                **{c: row[c] for c in (*FIGHT, "won", "final_hp", "potions", "terminal_value")},
                "turns": 0, "decisions": 0, "simulations": 0})
            fight["turns"] = max(fight["turns"], row["turn"])
            fight["decisions"] += 1
            fight["simulations"] += row["simulations_used"]
    for fight in result.values():
        fight["sims_per_decision"] = fight["simulations"] / fight["decisions"]
    return result


def paired(base, cand, rng):
    """Means, mean paired difference (candidate - baseline) with bootstrap 95% CI, Wilcoxon p."""
    diff = cand - base
    means = diff[rng.integers(0, len(diff), (RESAMPLES, len(diff)))].mean(axis=1)
    p = stats.wilcoxon(diff).pvalue if np.any(diff) else 1.0
    return {"baseline_mean": base.mean(), "candidate_mean": cand.mean(), "diff": diff.mean(),
            "ci_low": np.quantile(means, 0.025), "ci_high": np.quantile(means, 0.975), "p": p}


def compare(config):
    baseline = run_ids(config["run"]["baseline"], "baseline")
    candidate = run_ids(config["run"]["candidate"], "candidate")
    cand = fights(candidate)
    base = fights(baseline, cand.keys())
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
    summary = {"schema": NAME, "config": config, "n": len(episodes), "won": wins, **metrics,
               PRIMARY + "_better_worse_equal": [int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())]}
    pairs = pa.Table.from_pylist([
        {"episode_id": e, "encounter": cand[e]["encounter"],
         **{f"{side}_{m}": s[e][m] for m in ("won", *TESTED, *DESCRIBED) for side, s in (("baseline", base), ("candidate", cand))}}
        for e in episodes])
    return summary, pairs, base, cand


def report(summary, base, cand):
    config, n = summary["config"]["run"], summary["n"]
    inputs = {k: [json.loads((run_dir(run_id) / "run.json").read_text())["inputs"] for run_id in run_ids(config[k], k)]
              for k in ("baseline", "candidate")}
    t = summary[PRIMARY]
    better, worse, equal = summary[PRIMARY + "_better_worse_equal"]
    w = summary["won"]
    lines = [
        f"# {', '.join(run_ids(config['candidate'], 'candidate'))} vs {', '.join(run_ids(config['baseline'], 'baseline'))}", "",
        f"- baseline:  `{', '.join(run_ids(config['baseline'], 'baseline'))}` (inputs: {inputs['baseline'] or 'none'})",
        f"- candidate: `{', '.join(run_ids(config['candidate'], 'candidate'))}` (inputs: {inputs['candidate'] or 'none'})",
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


def main(config_path, out):
    config = tomllib.loads(config_path.read_text())
    summary, pairs, base, cand = compare(config)
    text = report(summary, base, cand)
    (out / "report.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    pq.write_table(pairs, out / "pairs.parquet")
    print(text, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, help="run output dir")
    parser.add_argument("--inputs", action="store_true", help="print the baseline and candidate run ids (for run.sh)")
    args = parser.parse_args()
    if args.inputs:
        run = tomllib.loads(args.config.read_text())["run"]
        print(*run_ids(run["baseline"], "baseline"), *run_ids(run["candidate"], "candidate"), sep="\n")
    else:
        main(args.config, args.out)
