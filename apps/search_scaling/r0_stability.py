"""R0: how much the reference disagrees with itself, and what is at stake per decision.

Usage: r0_stability.py REFERENCE_DIR
"""
import json, sys
from pathlib import Path
import numpy as np


def q_by_action(draw):
    return {e['bits']: e['q'] for e in draw['edges'] if e['visits'] > 0}


def main():
    rows = [json.loads(l) for l in open(Path(sys.argv[1]) / 'reference.jsonl')]
    self_regret, agree, gap, prior_regret, sd = [], [], [], [], []
    for row in rows:
        a, b = row['draws']
        qa, qb = q_by_action(a), q_by_action(b)
        common = sorted(qa.keys() & qb.keys())
        qref = {k: (qa[k] + qb[k]) / 2 for k in common}
        best = max(qref.values())
        ordered = sorted(qref.values(), reverse=True)
        gap.append(ordered[0] - ordered[1] if len(ordered) > 1 else 0.0)
        # A's move judged by B and B's move judged by A: one draw against an independent one.
        self_regret.append((max(qb.values()) - qb[a['action']] + max(qa.values()) - qa[b['action']]) / 2)
        agree.append(a['action'] == b['action'])
        prior_move = max(a['edges'], key=lambda e: e['logit'])['bits']
        prior_regret.append(best - qref[prior_move] if prior_move in qref else np.nan)
        sd += [abs(qa[k] - qb[k]) / np.sqrt(2) for k in common]
    n = len(rows)
    se = lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(~np.isnan(x)))
    self_regret, prior_regret, gap = map(np.array, (self_regret, prior_regret, gap))
    print(f'positions {n}')
    print(f'reference Q SD per action (one draw, from |A-B|/sqrt2): median {np.median(sd):.4f}  p90 {np.quantile(sd, .9):.4f}')
    print(f'move agreement A vs B: {np.mean(agree):.1%}')
    print(f'regret, one reference draw vs the other: mean {self_regret.mean():.5f} +- {se(self_regret):.5f}')
    print(f'regret, network prior alone (0 sims) vs averaged reference: mean {np.nanmean(prior_regret):.5f} +- {se(prior_regret):.5f}'
          f'  (zero in {np.nanmean(prior_regret == 0):.0%})')
    print('stakes: gap between best and second move (averaged reference)')
    for t in (0.001, 0.01, 0.05, 0.1):
        print(f'  gap > {t:<5}: {np.mean(gap > t):5.1%}')
    print(f'  median {np.median(gap):.4f}, mean {gap.mean():.4f}')
    print('prior regret by stakes (share of total prior regret)')
    tot = np.nansum(prior_regret)
    for lo, hi in ((0, .01), (.01, .05), (.05, 1.01)):
        m = (gap >= lo) & (gap < hi)
        print(f'  gap [{lo}, {hi}): {m.mean():5.1%} of positions carry {np.nansum(prior_regret[m]) / tot:5.1%} of prior regret')


if __name__ == '__main__':
    main()
