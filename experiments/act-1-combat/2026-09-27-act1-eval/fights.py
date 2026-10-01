#!/usr/bin/env python3
"""Per-fight breakdown of the two act1-eval runs: deaths and HP loss by fight category, over all fights played and
over matched fights (identical start in both runs: seed, fight_index, encounter, floor, start HP, max HP).
  PYTHONPATH=. .venv/bin/python experiments/act-1-combat/2026-09-27-act1-eval/fights.py
"""
from runs.query import connect
db = connect()
A, B = "combat_v3/2026-09-27/act1-eval-mcts-a20", "combat_v3/2026-09-27/act1-eval-ab-gen1-a20"
db.execute(f"""create temp table f as
  select case run_id when '{A}' then 'mcts' else 'net' end as agent, run_seed, fight_index, floor, category, encounter,
         any_value(starting_hp) as start_hp, any_value(starting_max_hp) as max_hp,
         any_value(won) as won, any_value(final_hp) as final_hp
  from combat_v3 where run_id in ('{A}', '{B}') and row_kind = 'decision' group by all""")
db.execute("create temp view g as select *, start_hp - final_hp as loss from f")
cats = "case category when 'easy' then 1 when 'hard' then 2 when 'event' then 3 when 'elite' then 4 else 5 end"
print("ALL FIGHTS PLAYED")
print(db.sql(f"""
  select category,
    count(*) filter (agent='mcts') as fights_mcts, count(*) filter (agent='net') as fights_net,
    count(*) filter (agent='mcts' and not won) as deaths_mcts, count(*) filter (agent='net' and not won) as deaths_net,
    round(100*avg((not won)::int) filter (agent='mcts'),2) as death_pct_mcts, round(100*avg((not won)::int) filter (agent='net'),2) as death_pct_net,
    round(avg(loss) filter (agent='mcts' and won),2) as loss_won_mcts, round(avg(loss) filter (agent='net' and won),2) as loss_won_net,
    round(stddev_samp(loss) filter (agent='mcts' and won)/sqrt(count(*) filter (agent='mcts' and won)),2) as se_m,
    round(stddev_samp(loss) filter (agent='net' and won)/sqrt(count(*) filter (agent='net' and won)),2) as se_n
  from g group by category order by any_value({cats})"""))
print("MATCHED FIGHTS: same seed, fight_index, encounter, floor, start hp, max hp")
db.execute("""create temp table m as select a.category, a.encounter, a.run_seed, a.fight_index,
   a.won::int as a_won, b.won::int as b_won, a.loss as a_loss, b.loss as b_loss
   from g a join g b using (run_seed, fight_index, encounter, floor, start_hp, max_hp)
   where a.agent='mcts' and b.agent='net'""")
q = lambda by: db.sql(f"""
  select {by}, count(*) as n,
    sum(1-a_won) as deaths_mcts, sum(1-b_won) as deaths_net,
    round(avg(a_loss),2) as loss_mcts, round(avg(b_loss),2) as loss_net,
    round(avg(b_loss-a_loss),2) as diff, round(stddev_samp(b_loss-a_loss)/sqrt(count(*)),2) as se
  from m group by {by} order by {"any_value("+cats+")" if by=="category" else "category, n desc"}""")
print("(loss counts a death as its starting HP)")
print(q("category"))
print(q("category, encounter").filter("category in ('elite','boss')"))
print("fight index where runs first diverge", db.sql("select fight_index, count(*) from (select run_seed, min(fight_index) fight_index from g a where not exists (select 1 from m where m.run_seed=a.run_seed and m.fight_index=a.fight_index) group by 1) group by 1 order by 1").fetchall())
print("deaths outside combat:", db.sql("""select r.run_id, count(*) from act1_results r where run_id like '%act1-eval-%a20' and status='died'
   and not exists (select 1 from g where g.run_seed=r.seed and not g.won and g.agent = case when r.run_id like '%mcts%' then 'mcts' else 'net' end) group by 1""").fetchall())
print("\nMARKDOWN")
for r in db.sql(f"""
  select category,
    count(*) filter (agent='mcts'), count(*) filter (agent='net'),
    count(*) filter (agent='mcts' and not won), count(*) filter (agent='net' and not won),
    100*avg((not won)::int) filter (agent='mcts'), 100*avg((not won)::int) filter (agent='net'),
    avg(start_hp) filter (agent='mcts'), avg(start_hp) filter (agent='net')
  from g group by category order by any_value({cats})""").fetchall():
    c, fm, fn, dm, dn, pm, pn, hm, hn = r
    se = (pm*(100-pm)/fm + pn*(100-pn)/fn) ** .5
    print(f"| {c} | {fm} / {fn} | {dm} / {dn} | {pm:.1f}% / {pn:.1f}% | {pn-pm:+.1f} ±{se:.1f} | {hm:.1f} / {hn:.1f} |")
