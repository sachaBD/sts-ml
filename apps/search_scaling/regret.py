"""Regret of each grid cell against the averaged reference, with paired differences.

Usage: regret.py REFERENCE_DIR GRID_DIR
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np


def reference(ref):
    """Per position: averaged reference Q and target by action, and its best Q."""
    out = []
    for line in open(ref / 'reference.jsonl'):
        a, b = json.loads(line)['draws']
        qa = {e['bits']: e['q'] for e in a['edges'] if e['visits'] > 0}
        qb = {e['bits']: e['q'] for e in b['edges'] if e['visits'] > 0}
        q = {k: (qa[k] + qb[k]) / 2 for k in qa.keys() & qb.keys()}
        target = {e['bits']: e['improved_policy'] / 2 for e in a['edges']}
        for e in b['edges']:
            target[e['bits']] += e['improved_policy'] / 2
        prior = max(a['edges'], key=lambda e: e['logit'])['bits']
        out.append(dict(q=q, best=max(q.values()), target=target, prior=prior,
                        prior_policy={e['bits']: e['prior'] for e in a['edges']}))
    return out


def main():
    ref, grid = Path(sys.argv[1]), Path(sys.argv[2])
    refs = reference(ref)
    regret, tv, requests = defaultdict(dict), defaultdict(dict), defaultdict(list)
    for line in open(grid / 'cells.jsonl'):
        c = json.loads(line)
        r, cell = refs[c['position']], (c['sims'], c['worlds'])
        # A move the reference never visited in both draws is valued at its worst visited move.
        regret[cell][c['position']] = r['best'] - r['q'].get(c['action'], min(r['q'].values()))
        tv[cell][c['position']] = 0.5 * sum(abs(e['improved_policy'] - r['target'][e['bits']]) for e in c['edges'])
        requests[cell].append(c['network_requests'])
    for k in {k for cell in regret.values() for k in cell}:  # the network alone: 0 sims
        r = refs[k]
        regret[(0, 0)][k] = r['best'] - r['q'].get(r['prior'], min(r['q'].values()))
        tv[(0, 0)][k] = 0.5 * sum(abs(p - r['target'][b]) for b, p in r['prior_policy'].items())
        requests[(0, 0)].append(1)
    cells = sorted(regret)
    n = len(regret[cells[0]])
    print(f'positions {n}; regret in win points per decision (mean +- SE)')
    print(f'{"sims":>5} {"worlds":>6} {"requests":>9} {"regret":>9} {"SE":>8} {"TV target":>10}')
    for cell in cells:
        x = np.array(list(regret[cell].values()))
        print(f'{cell[0]:5} {cell[1]:6} {np.mean(requests[cell]):9.0f} {x.mean():9.5f} {x.std(ddof=1) / np.sqrt(n):8.5f} '
              f'{np.mean(list(tv[cell].values())):10.3f}')
    base = (256, 8)
    if base in regret:
        print(f'\npaired difference vs {base}: mean, SD of the difference, n for SE 0.00025')
        for cell in cells:
            if cell == base:
                continue
            d = np.array([regret[cell][k] - regret[base][k] for k in regret[base]])
            sd = d.std(ddof=1)
            print(f'{cell[0]:5} {cell[1]:6}  {d.mean():+9.5f}  SD {sd:.5f}  n {int(np.ceil((sd / 0.00025) ** 2)):6d}')


if __name__ == '__main__':
    main()
