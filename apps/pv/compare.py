#!/usr/bin/env python3
"""Paired comparison of fight outputs (combat_v4 `fights-*.parquet` dirs, e.g. from apps/pv/play.py) on fight_id.

  compare.py A_OUT B_OUT [C_OUT ...] [--baseline A_OUT] [--json]

Each arm is compared with the baseline (default: the first dir) on the fight_ids both have; n is printed per arm.
Score = HP-equivalent points: 0 for a loss, 35 + final_hp for a win. Potions left at the end are not stored in
`fights` (only the start is), so the +4/potion term is omitted. Diffs are arm - baseline, ± 1 SE of the paired
differences. Discordant: b = baseline won / arm lost, c = baseline lost / arm won; p is the exact two-sided McNemar test.
Splits: Demon Form in the start deck (CardId 107) and start HP band. Search cost, when decisions_stats-*.parquet
exists: seconds/fight, visit-weighted mean depth, and p50/p90 of the per-decision maximum depth (actions / player turns).
"""
import argparse
import json
from math import comb, sqrt
from pathlib import Path

import duckdb

DEMON_FORM = 107
BANDS = [('hp<40', 'a.hp < 40'), ('hp 40-55', 'a.hp >= 40 and a.hp < 55'), ('hp 55-70', 'a.hp >= 55 and a.hp < 70'), ('hp>=70', 'a.hp >= 70')]
SPLITS = [('all', 'true'), ('demon form', 'a.demon'), ('no demon form', 'not a.demon'), *BANDS]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def load(db, name, path):
    db.execute(f"""create table {name} as select fight_id, won, final_hp, start.hp as hp,
        list_contains(list_transform(start.deck, c -> c.id), {DEMON_FORM}) as demon,
        case when won then 35 + final_hp else 0 end as score from read_parquet('{path}/fights-*.parquet')""")
    if db.execute(f'select count(*) - count(distinct fight_id) from {name}').fetchone()[0]:
        raise ValueError(f'{path}: duplicate fight_id')


def cost(db, path, fights):
    files = list(Path(path).glob('decisions_stats-*.parquet'))
    if not files: return None
    s = f"read_parquet('{path}/decisions_stats-*.parquet')"
    r = db.execute(f"""select sum(seconds) / {fights}, sum(mean_depth * simulations) / sum(simulations),
        sum(mean_turns * simulations) / sum(simulations), quantile_cont(max_depth, 0.5), quantile_cont(max_depth, 0.9),
        quantile_cont(max_turns, 0.5), quantile_cont(max_turns, 0.9) from {s}""").fetchone()
    return dict(zip(['seconds_per_fight', 'mean_depth', 'mean_turns', 'depth_max_p50', 'depth_max_p90', 'turns_max_p50', 'turns_max_p90'], r))


def compare(base, arm):
    out = []
    for label, where in SPLITS:
        n, bw, aw, diff, sd, b, c = db.execute(f"""select count(*), avg(a.won::int), avg(x.won::int), avg(x.score - a.score),
            stddev_samp(x.score - a.score), sum((a.won and not x.won)::int), sum((not a.won and x.won)::int)
            from {base} a join {arm} x using (fight_id) where {where}""").fetchone()
        out.append(dict(split=label, n=n, base_win=bw, arm_win=aw, score_diff=diff,
                        se=(sd / sqrt(n) if n > 1 and sd is not None else None), b=b or 0, c=c or 0, p=mcnemar(b or 0, c or 0)))
    return out


def fmt(x, spec='.3f'):
    return '-' if x is None else format(x, spec)


db = duckdb.connect()
db.execute("set memory_limit='2GB'; set threads=1")

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('arms', type=Path, nargs='+')
    p.add_argument('--baseline', type=Path)
    p.add_argument('--json', action='store_true')
    a = p.parse_args()
    paths = a.arms if a.baseline is None or a.baseline in a.arms else [a.baseline, *a.arms]
    base_path = a.baseline or a.arms[0]
    names = {path: f'arm{i}' for i, path in enumerate(paths)}
    for path, name in names.items(): load(db, name, path)
    report = {}
    for path in paths:
        n = db.execute(f'select count(*) from {names[path]}').fetchone()[0]
        report[str(path)] = dict(fights=n, cost=cost(db, path, n),
                                 vs_baseline=None if path == base_path else compare(names[base_path], names[path]))
    if a.json:
        print(json.dumps(report, indent=1))
    else:
        print(f'baseline: {base_path}\n\n| arm | fights | s/fight | depth (actions) | depth (turns) | depth_max p50/p90 | turns_max p50/p90 |\n|---|---|---|---|---|---|---|')
        for path, r in report.items():
            c = r['cost'] or {}
            print(f"| {path} | {r['fights']} | {fmt(c.get('seconds_per_fight'), '.2f')} | {fmt(c.get('mean_depth'), '.2f')} | {fmt(c.get('mean_turns'), '.2f')} | "
                  f"{fmt(c.get('depth_max_p50'), '.0f')}/{fmt(c.get('depth_max_p90'), '.0f')} | {fmt(c.get('turns_max_p50'), '.0f')}/{fmt(c.get('turns_max_p90'), '.0f')} |")
        for path, r in report.items():
            if r['vs_baseline'] is None: continue
            print(f'\n**{path} vs baseline** (score = 0 loss / 35 + final_hp win; diff = arm - baseline ± 1 SE)\n\n'
                  '| split | n | base win | arm win | score diff ± SE | b / c | McNemar p |\n|---|---|---|---|---|---|---|')
            for s in r['vs_baseline']:
                print(f"| {s['split']} | {s['n']} | {fmt(s['base_win'])} | {fmt(s['arm_win'])} | {fmt(s['score_diff'], '+.2f')} ± {fmt(s['se'], '.2f')} | {s['b']} / {s['c']} | {s['p']:.3f} |")
