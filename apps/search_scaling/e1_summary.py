"""E1 summary: paired regret differences vs 256 x 8, regret by stakes, and a per-fight scale.

Usage: e1_summary.py REFERENCE_DIR GRID_DIR
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps.search_scaling.regret import reference

BASE = (256, 8)
COMPARE = [(0, 0), (256, 1), (256, 16), (256, 64), (512, 8), (512, 16), (1024, 8), (1024, 16), (1024, 64)]


def main():
    ref, grid = Path(sys.argv[1]), Path(sys.argv[2])
    refs = reference(ref)
    positions = [json.loads(l) for l in open(ref / 'positions.jsonl')]
    regret = {}
    for line in open(grid / 'cells.jsonl'):
        c = json.loads(line)
        r = refs[c['position']]
        regret.setdefault((c['sims'], c['worlds']), {})[c['position']] = r['best'] - r['q'].get(c['action'], min(r['q'].values()))
    keys = sorted(regret[BASE])
    regret[(0, 0)] = {k: refs[k]['best'] - refs[k]['q'].get(refs[k]['prior'], min(refs[k]['q'].values())) for k in keys}
    per_fight = np.mean([p['decisions'] for p in positions])
    print(f'positions {len(keys)}; decisions with 2+ actions per fight: mean {per_fight:.1f}')
    print(f'\npaired difference vs {BASE} (negative = better), win points per decision, and x decisions per fight')
    base = np.array([regret[BASE][k] for k in keys])
    for cell in COMPARE:
        d = np.array([regret[cell][k] for k in keys]) - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f'{cell[0]:5} x {cell[1]:<3} {d.mean():+.5f} +- {se:.5f}  (z {d.mean() / se:+5.1f})   per fight {d.mean() * per_fight:+.4f}')
    gaps = []
    for k in keys:
        q = sorted(refs[k]['q'].values(), reverse=True)
        gaps.append(q[0] - q[1] if len(q) > 1 else 0)
    gaps = np.array(gaps)
    bands = [(0, .01), (.01, .05), (.05, 1.01)]
    print('\nregret by stakes (gap between the reference\'s best two moves); share of positions in brackets')
    print('cell        ' + ''.join(f'  gap [{lo}, {hi})' for lo, hi in bands))
    print(' ' * 12 + ''.join(f'  {np.mean((gaps >= lo) & (gaps < hi)):>13.1%}' for lo, hi in bands))
    for cell in [(0, 0), (64, 8), BASE, (256, 16), (1024, 1), (1024, 16), (1024, 64)]:
        x = np.array([regret[cell][k] for k in keys])
        print(f'{cell[0]:5} x {cell[1]:<3} ' + ''.join(f'  {x[(gaps >= lo) & (gaps < hi)].mean():13.5f}' for lo, hi in bands))


if __name__ == '__main__':
    main()
