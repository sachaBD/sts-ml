#!/usr/bin/env python3
"""Paired comparison of two combat_v3 runs on the same fights (slime-v8 reports).

.venv/bin/python rundecks/slime-v8/compare.py BASELINE_RUN_ID CANDIDATE_RUN_ID [--json OUT]
Per fight (episode_id): terminal_value, won, final_hp of the decision-0 row. Checks both runs cover the same fights
with the same starting HP. Mean terminal-value difference (candidate - baseline) with a paired percentile bootstrap
95% CI over fights (20,000 resamples, seed 0); wins, fights won by only one run, exact McNemar p.
"""
import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from sts_combat_rl.query import connect  # noqa: E402


def fights(db, run_id):
    rows = db.sql(f"""select episode_id, starting_hp, terminal_value, won, final_hp from combat_v3
                      where run_id = '{run_id}' and row_kind = 'decision' and decision_index = 0""").fetchall()
    if not rows:
        sys.exit(f"{run_id}: no fights")
    return {r[0]: r[1:] for r in rows}


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    db = connect()
    base, cand = fights(db, args.baseline), fights(db, args.candidate)
    if set(base) != set(cand):
        sys.exit(f"different fights: {len(set(base) ^ set(cand))} not in both")
    episodes = sorted(base)
    if bad := [e for e in episodes if base[e][0] != cand[e][0]]:
        sys.exit(f"{len(bad)} fights start at different HP, e.g. {bad[:3]}")
    tb = np.array([base[e][1] for e in episodes])
    tc = np.array([cand[e][1] for e in episodes])
    wb = np.array([base[e][2] for e in episodes], dtype=bool)
    wc = np.array([cand[e][2] for e in episodes], dtype=bool)
    diff = tc - tb
    rng = np.random.default_rng(0)
    boot = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(axis=1)
    only_c, only_b = int((wc & ~wb).sum()), int((wb & ~wc).sum())
    result = {
        "baseline": args.baseline, "candidate": args.candidate, "fights": len(episodes),
        "baseline_wins": int(wb.sum()), "candidate_wins": int(wc.sum()),
        "baseline_terminal_value": float(tb.mean()), "candidate_terminal_value": float(tc.mean()),
        "difference": float(diff.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "wins_only_candidate": only_c, "wins_only_baseline": only_b, "mcnemar_p": mcnemar(only_c, only_b),
        "identical_terminal_value_fights": int((diff == 0).sum()),
    }
    print(f"fights {result['fights']}")
    print(f"{'':12}{'wins':>6}{'terminal value':>16}")
    print(f"{'baseline':12}{result['baseline_wins']:>6}{result['baseline_terminal_value']:>16.4f}   {args.baseline}")
    print(f"{'candidate':12}{result['candidate_wins']:>6}{result['candidate_terminal_value']:>16.4f}   {args.candidate}")
    print(f"difference (candidate - baseline): {result['difference']:+.4f}  95% CI "
          f"({result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f})")
    print(f"wins only candidate / only baseline: {only_c} / {only_b}  (McNemar p {result['mcnemar_p']:.3f})")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
