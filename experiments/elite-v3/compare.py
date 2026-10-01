#!/usr/bin/env python3
"""Paired comparison of value_play runs on the same fights: each candidate against one baseline.

    PYTHONPATH=. .venv/bin/python experiments/elite-v3/compare.py --baseline ID --candidate ID [--candidate ID ...]
        [--title TEXT] [--output FILE.md] [--json FILE.json]

Runs are named by id (the combat_v3 view's `id`, unique across dates for this rundeck). Fights are paired on
episode_id and must share the start (run seed, encounter, starting HP, max HP); a fight missing from either run
(e.g. skipped because its replay diverged from the stored fight) is left out and counted.

HP-eq = (candidate terminal value - baseline terminal value) * (55 + max HP): the score difference in HP units
(a win at the same HP and potions = 0; one potion = 4). 95% intervals: 10,000 cluster bootstraps over source run
seeds (fights of one run share a deck), percentile. Per elite and overall.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from runs.query import connect

ELITES = ("gremlin_nob", "lagavulin", "three_sentries")


def fights(db, run_id: str) -> dict[int, dict]:
    rows = db.sql(f"""
        select episode_id, run_seed, encounter, starting_hp, starting_max_hp, won, final_hp, potions, terminal_value
        from combat_v3 where id = '{run_id}' and row_kind = 'decision' and decision_index = 0""").fetchall()
    keys = ("episode_id", "run_seed", "encounter", "starting_hp", "starting_max_hp", "won", "final_hp", "potions",
            "terminal_value")
    out = {r[0]: dict(zip(keys, r)) for r in rows}
    if not out:
        raise SystemExit(f"no fights in run id {run_id!r}")
    return out


def table(base: dict, cand: dict, rng) -> tuple[list[str], dict]:
    common = sorted(set(base) & set(cand))
    pairs = []
    for e in common:
        b, c = base[e], cand[e]
        if (b["run_seed"], b["encounter"], b["starting_hp"], b["starting_max_hp"]) != \
                (c["run_seed"], c["encounter"], c["starting_hp"], c["starting_max_hp"]):
            raise SystemExit(f"episode {e}: the runs did not start the same fight")
        pairs.append((b, c))
    lines, summary = [], {}
    for enc in (*ELITES, "all"):
        v = [(b, c) for b, c in pairs if enc == "all" or b["encounter"] == enc]
        if not v:
            continue
        diffs = np.array([(c["terminal_value"] - b["terminal_value"]) * (55 + b["starting_max_hp"]) for b, c in v])
        groups = defaultdict(list)
        for (b, _), d in zip(v, diffs):
            groups[b["run_seed"]].append(d)
        sums = np.array([sum(g) for g in groups.values()])
        counts = np.array([len(g) for g in groups.values()])
        idx = rng.integers(0, len(sums), (10000, len(sums)))
        lo, hi = np.quantile(sums[idx].sum(axis=1) / counts[idx].sum(axis=1), [0.025, 0.975])
        wb, wc = sum(b["won"] for b, _ in v), sum(c["won"] for _, c in v)
        pb = np.mean([b["potions"] for b, _ in v if b["won"]] or [0])
        pc = np.mean([c["potions"] for _, c in v if c["won"]] or [0])
        only_c = sum(c["won"] and not b["won"] for b, c in v)
        only_b = sum(b["won"] and not c["won"] for b, c in v)
        lines.append(f"| {enc} | {len(v)} | {diffs.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {wb} / {wc} | "
                     f"{only_c} / {only_b} | {pb:.2f} / {pc:.2f} |")
        summary[enc] = {"fights": len(v), "hp_eq": float(diffs.mean()), "lo": float(lo), "hi": float(hi),
                        "wins_baseline": int(wb), "wins_candidate": int(wc)}
    summary["unpaired"] = {"baseline_only": len(set(base) - set(cand)), "candidate_only": len(set(cand) - set(base))}
    return lines, summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--title", default="Paired comparison")
    p.add_argument("--output", type=Path)
    p.add_argument("--json", type=Path, help="per candidate: {encounter: {fights, hp_eq, lo, hi, wins...}}")
    a = p.parse_args()
    db = connect()
    base = fights(db, a.baseline)
    rng = np.random.default_rng(2026)
    out = [f"# {a.title}", "", f"Baseline: `{a.baseline}`. Positive HP-eq = candidate better. 95% intervals: 10,000 "
           "source-run-seed cluster bootstraps (percentile).", ""]
    results = {}
    for cand_id in a.candidate:
        lines, summary = table(base, fights(db, cand_id), rng)
        results[cand_id] = summary
        out += [f"## `{cand_id}` vs baseline", "",
                "| Encounter | Fights | HP-eq cand − base [95% CI] | Wins base / cand | Cand-only / base-only wins "
                "| Potions kept on wins base / cand |", "|---|---:|---:|---:|---:|---:|", *lines, "",
                f"Unpaired fights (in one run only, e.g. diverged replays): baseline-only "
                f"{summary['unpaired']['baseline_only']}, candidate-only {summary['unpaired']['candidate_only']}.", ""]
    text = "\n".join(out)
    print(text)
    if a.output:
        a.output.write_text(text + "\n")
    if a.json:
        import json
        a.json.write_text(json.dumps({"baseline": a.baseline, "candidates": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
