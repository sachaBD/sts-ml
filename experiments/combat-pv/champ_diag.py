"""Champ fight diagnostics from combat_v4_full decisions (one or more decompressed out dirs).

  champ_diag.py OUT_DIR [OUT_DIR ...] [--by-fight F.parquet]

Per fight: won, turns, start HP, deck has Demon Form, turn Champ first stands below 50% HP (cross), player / Champ
Strength at the first decision of that turn, turns from cross to the end, Execute turns faced, scaling cards played
before the cross, debuff cards played while Champ's intent is Anger (Anger removes debuffs), player max Strength.
Aggregates by agent; prints mean ± 1 SE (n). Only decisions (not events between them) are seen: a value observed
"at the cross" is at the first decision of the cross turn.
"""
import argparse
import math

import duckdb

SCALING = ['demon_form', 'inflame', 'spot_weakness', 'limit_break', 'flex', 'rupture', 'metallicize', 'barricade',
           'feel_no_pain', 'juggernaut', 'dark_embrace', 'corruption', 'brutality', 'evolve', 'combust', 'fire_breathing']
DEBUFF = ['bash', 'uppercut', 'shockwave', 'thunderclap', 'clothesline', 'intimidate', 'trip', 'blind']


def per_fight(dirs):
    files = [f'{d}/decisions-*.parquet' for d in dirs]
    fights = [f'{d}/fights-*.parquet' for d in dirs]
    return duckdb.sql(f"""
with d as (
  select fight_id, step, public.turn turn, public.player p, public.hand hand, legal[chosen + 1].action a,
         list_filter(public.monsters, m -> m.monster = 'the_champ')[1] ch,
         list_concat(public.hand, public.draw, public.discard, public.exhaust) cards
  from read_parquet({files!r})),
x as (
  select *, coalesce(list_filter(ch.statuses, s -> s.status = 'strength')[1].amount, 0) champ_str,
         coalesce(list_filter(p.statuses, s -> s.status = 'strength')[1].amount, 0) player_str,
         case when (a >> 29) = 0 then hand[(a & 65535) + 1].card end played
  from d where ch is not null),
crossing as (select fight_id, min(turn) cross_turn from x where ch.hp * 2 < ch.max_hp group by 1),
at_cross as (
  select x.fight_id, arg_min(player_str, step) player_str_cross, arg_min(champ_str, step) champ_str_cross,
         arg_min(p.hp, step) hp_cross
  from x join crossing c on x.fight_id = c.fight_id and x.turn = c.cross_turn group by 1),
agg as (
  select x.fight_id, max(turn) last_turn, max(player_str) player_str_max,
         count(distinct turn) filter (where ch.intent = 'the_champ_execute') execute_turns,
         count(*) filter (where played in {tuple(SCALING)!r} and (c.cross_turn is null or turn < c.cross_turn)) scaling_pre,
         count(*) filter (where played in {tuple(DEBUFF)!r} and ch.intent = 'the_champ_anger') debuff_into_anger,
         bool_or(list_contains(list_transform(cards, c -> c.card), 'demon_form')) has_df,
         any_value(c.cross_turn) cross_turn
  from x left join crossing c using (fight_id) group by 1)
select f.agent, f.fight_id, f.won, f.turns, f.start_hp, f.final_hp, f.monster_hp_left, agg.*  exclude (fight_id),
       at_cross.* exclude (fight_id), agg.last_turn - agg.cross_turn turns_after_cross
from read_parquet({fights!r}) f join agg using (fight_id) left join at_cross using (fight_id)""")


def mean_se(xs):
    xs = [float(x) for x in xs if x is not None]
    n = len(xs)
    if n < 2:
        return f'n={n}'
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)
    return f'{m:.2f} ± {se:.2f} ({n})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--by-fight', help='write the per-fight table to this parquet')
    a = ap.parse_args()
    rel = per_fight(a.dirs)
    if a.by_fight:
        rel.write_parquet(a.by_fight)
    rows = rel.fetchall()
    cols = rel.columns
    groups = {}
    for r in rows:
        r = dict(zip(cols, r))
        for key in (r['agent'], f"{r['agent']} | {'DF' if r['has_df'] else 'no DF'}"):
            groups.setdefault(key, []).append(r)
    metrics = ['won', 'turns', 'start_hp', 'scaling_pre', 'player_str_max', 'execute_turns', 'debuff_into_anger']
    cross_metrics = ['cross_turn', 'player_str_cross', 'champ_str_cross', 'hp_cross', 'turns_after_cross']
    for key, rs in sorted(groups.items()):
        print(f'== {key}  n={len(rs)}')
        for m in metrics:
            print(f'  {m:20s} {mean_se([r[m] for r in rs])}')
        crossed = [r for r in rs if r['cross_turn'] is not None]
        print(f'  crossed 50%          {len(crossed)}/{len(rs)}; win | crossed {mean_se([r["won"] for r in crossed])}')
        for m in cross_metrics:
            print(f'  {m:20s} {mean_se([r[m] for r in crossed])}')
        burst = [r for r in crossed if r['won']]
        print(f'  wins: turns_after_cross {mean_se([r["turns_after_cross"] for r in burst])}')


if __name__ == '__main__':
    main()
