#!/usr/bin/env python3
"""Win rates at 20k vs 100k on the same rebuilt Act 2 boss fights; paired difference with 95% CI.

  analyze.py RUN_OUT            (prints a markdown summary)
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

LO, HI = 20000, 100000


def paired(rows):
    """rows: [(win_lo, win_hi)] -> (n, rate_lo, rate_hi, diff, se, only_lo, only_hi, exact McNemar p)."""
    n = len(rows)
    b = sum(1 for lo, hi in rows if lo and not hi)
    c = sum(1 for lo, hi in rows if hi and not lo)
    diff = (c - b) / n
    se = math.sqrt(max((b + c) / n - diff ** 2, 0) / n)
    k, m = min(b, c), b + c
    p = min(1.0, 2 * sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m) if m else 1.0
    return n, sum(lo for lo, _ in rows) / n, sum(hi for _, hi in rows) / n, diff, se, b, c, p


def main():
    out = Path(sys.argv[1])
    fights = {json.loads(l)['fight_id']: json.loads(l) for l in open(out / 'fights.jsonl')}
    res = defaultdict(dict)
    for line in open(out / 'results.jsonl'):
        r = json.loads(line)
        res[r['fight_id']][r['sims']] = r
    both = {k: v for k, v in res.items() if LO in v and HI in v}
    groups = defaultdict(list)
    for k, v in both.items():
        for g in ('all', fights[k]['encounter']):
            groups[g].append(k)

    print(f'| Boss | n | 20k win | 100k win | 100k − 20k (95% CI) | 20k-only / 100k-only wins | McNemar p |')
    print('|---|---:|---:|---:|---|---:|---:|')
    for g in ('all', 'automaton', 'champ', 'collector'):
        ks = groups[g]
        n, a, b, d, se, only_lo, only_hi, p = paired([(both[k][LO]['won'], both[k][HI]['won']) for k in ks])
        print(f'| {g} | {n} | {a:.1%} | {b:.1%} | {d * 100:+.1f} pp ({(d - 1.96 * se) * 100:+.1f}, '
              f'{(d + 1.96 * se) * 100:+.1f}) | {only_lo} / {only_hi} | {p:.2f} |')

    print('\nFidelity (rebuilt 20k vs original recorded 20k, same fights):')
    for g in ('all', 'automaton', 'champ', 'collector'):
        ks = groups[g]
        n, a, b, d, se, only_o, only_r, p = paired([(fights[k]['orig_won'], both[k][LO]['won']) for k in ks])
        print(f'  {g}: original {a:.1%}, rebuilt {b:.1%}, diff {d * 100:+.1f} ± {se * 100:.1f} pp (1 SE), '
              f'disagree {only_o + only_r}/{n}')

    print('\nPer arm (means over fights):')
    for s in (LO, HI):
        for g in ('all', 'automaton', 'champ', 'collector'):
            rs = [both[k][s] for k in groups[g]]
            won = [r for r in rs if r['won']]
            lost = [r for r in rs if not r['won']]
            held = [r for r in rs if r['potions_start'] > 0]
            print(f'  {s // 1000}k {g}: sec/fight {sum(r["seconds"] for r in rs) / len(rs):.1f}, '
                  f'turns {sum(r["turns"] for r in rs) / len(rs):.1f}, '
                  f'HP left on win {sum(r["final_hp"] for r in won) / max(len(won), 1):.1f}, '
                  f'boss HP left on loss {sum(r["monster_hp_left"] / r["monster_max_hp_start"] for r in lost) / max(len(lost), 1):.0%}, '
                  f'potions used/held {sum(r["potions_start"] - r["potions_end"] for r in held) / max(sum(r["potions_start"] for r in held), 1):.0%}, '
                  f'died holding potion {sum(1 for r in lost if r["potions_end"] > 0)}/{len(lost)}')


if __name__ == '__main__':
    main()
