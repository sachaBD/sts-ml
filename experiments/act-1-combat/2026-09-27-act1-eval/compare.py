#!/usr/bin/env python3
"""Paired full-act-1 comparison of two apps/bootstrap runs on the same seeds (runs.query act1_results).

  PYTHONPATH=. .venv/bin/python experiments/act-1-combat/2026-09-27-act1-eval/compare.py [RUN_A RUN_B]

Uncertainty: +- is one standard error; for differences, of the per-seed paired difference.
"""
import sys

from runs.query import connect

A, B = sys.argv[1:3] if len(sys.argv) == 3 else ("combat_v3/2026-09-27/act1-eval-mcts-a20",
                                                   "combat_v3/2026-09-27/act1-eval-ab-gen1-a20")
db = connect()
paired = f"""
    select a.seed, a.boss,
           a.won_boss::int as a_clear, b.won_boss::int as b_clear,
           a.reached_boss::int as a_reach, b.reached_boss::int as b_reach,
           a.final_hp as a_hp, b.final_hp as b_hp, a.floor as a_floor, b.floor as b_floor
    from (select * from act1_results where run_id = '{A}' and status != 'other_boss') a
    join (select * from act1_results where run_id = '{B}' and status != 'other_boss') b using (seed)"""


def stat(label, a, b, where="true"):
    n, ma, mb, sa, sb, md, sd = db.sql(f"""
        select count(*), avg({a}), avg({b}), stddev_samp({a}) / sqrt(count(*)), stddev_samp({b}) / sqrt(count(*)),
               avg({b} - {a}), stddev_samp({b} - {a}) / sqrt(count(*))
        from ({paired}) where {where}""").fetchone()
    if n < 2:
        print(f"{label:36}{n:>6}  (too few seeds)")
        return
    print(f"{label:36}{n:>6}  {ma:>7.3f} ±{sa:.3f}  {mb:>7.3f} ±{sb:.3f}  {md:>+7.3f} ±{sd:.3f}")


print(f"A = {A}\nB = {B}\n")
print(f"{'':36}{'seeds':>6}  {'A':>14}  {'B':>14}  {'B - A':>14}")
stat("act cleared", "a_clear", "b_clear")
stat("reached boss", "a_reach", "b_reach")
stat("final HP (0 if died)", "a_hp", "b_hp")
stat("floor reached", "a_floor", "b_floor")
for boss in ("slime_boss", "the_guardian", "hexaghost"):
    stat(f"cleared | {boss}", "a_clear", "b_clear", f"boss = '{boss}'")
    stat(f"boss won | both reached {boss}", "a_clear", "b_clear", f"boss = '{boss}' and a_reach = 1 and b_reach = 1")

print("\nclear outcome (seeds): A cleared / B cleared")
print(db.sql(f"select a_clear as a, b_clear as b, count(*) as seeds from ({paired}) group by all order by all").fetchall())

print("\ndeaths by encounter (last fight of each died run)")
db.sql(f"""
    with died as (select run_id, seed from act1_results where run_id in ('{A}', '{B}') and status = 'died'),
    last as (select run_id, run_seed, arg_max(encounter, fight_index) as encounter, arg_max(category, fight_index) as category
             from combat_v3 where run_id in ('{A}', '{B}') and row_kind = 'decision' group by all)
    select category, encounter, count(*) filter (where d.run_id = '{A}') as a,
           count(*) filter (where d.run_id = '{B}') as b
    from died d join last l on l.run_id = d.run_id and l.run_seed = d.seed
    group by all order by a + b desc""").show(max_rows=40)
