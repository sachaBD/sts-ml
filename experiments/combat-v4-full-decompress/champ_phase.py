"""Per Champ fight: the turn Champ's phase2 counter (enrage) first shows, when it first stood below 50% HP, and how
long the player survived after. Reads a combat_v4_full run's out dir.

  .venv/bin/python experiments/combat-v4-full-decompress/champ_phase.py runs/schema=combat_v4_full/date=.../id=ah-champ/out
"""
import sys

import duckdb

out = sys.argv[1]
db = duckdb.connect()
print(db.execute(f"""
with ch as (
  select fight_id, public.turn as turn, m.hp as hp, m.max_hp as max_hp,
         coalesce(list_filter(m.counters, x -> x.status = 'phase2')[1].amount, 0) as phase2,
         coalesce(list_filter(m.statuses, x -> x.status = 'strength')[1].amount, 0) as strength
  from (select *, unnest(public.monsters) as m from read_parquet('{out}/decisions-*.parquet'))
  where m.monster = 'the_champ'),
per as (
  select fight_id,
         min(turn) filter (where hp * 2 < max_hp) as below_half_turn,
         min(turn) filter (where phase2 = 1) as phase2_turn,
         max(strength) as max_strength, max(turn) as last_turn
  from ch group by 1)
select f.fight_id, f.won, f.turns, f.start_hp, f.final_hp, below_half_turn, phase2_turn, last_turn - phase2_turn as turns_after_phase2, max_strength
from per join read_parquet('{out}/fights-*.parquet') f using (fight_id)
order by f.fight_id""").fetchall())
