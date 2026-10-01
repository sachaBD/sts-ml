#!/usr/bin/env python3
"""Count eligible terminal-label decision rows (exclude forced-random and earlier baseline rows)."""
from pathlib import Path
from sts_combat_rl.query import connect

HERE = Path(__file__).parent
base = 'combat_v3/2026-09-26/act1-all-bosses-a20-scaled-search'
selfplay = 'combat_v3/2026-09-27/ab-selfplay-gen1-rest'
db = connect()
q = f"""
with decisions as (
  select run_id, run_seed, episode_id, source_episode_id, encounter, decision_index, was_random
  from combat_v3
  where run_id in ('{base}', '{selfplay}') and category = 'elite' and row_kind = 'decision'
), fights as (
  select run_id, run_seed, episode_id, any_value(source_episode_id) source_episode_id,
         any_value(encounter) encounter, count(*) decision_rows,
         max(decision_index) filter (where was_random) random_at,
         count(*) filter (where was_random) random_rows
  from decisions group by run_id, run_seed, episode_id
), eligible as (
  select d.run_id, d.run_seed, d.episode_id,
         count(*) filter (where f.random_at is null or d.decision_index > f.random_at) terminal_rows
  from decisions d join fights f using (run_id, run_seed, episode_id) group by all
)
select case when f.run_id = '{selfplay}' then 'train replay bucket 4'
            when f.run_seed % 10 in (2,3) then 'final 2-3'
            when f.run_seed % 10 = 5 then 'validation 5'
            when f.run_seed % 10 in (0,1) then 'excluded 0-1'
            else 'train baseline 4,6-9' end part,
       f.encounter, count(*) fights, sum(f.decision_rows) decisions,
       sum(e.terminal_rows) terminal_eligible, count(*) filter (where e.terminal_rows = 0) no_terminal_rows,
       count(*) filter (where f.random_rows > 0) fights_with_random,
       count(*) filter (where f.random_rows > 1) fights_multi_random
from fights f join eligible e using (run_id, run_seed, episode_id)
group by all order by part, encounter
"""
rel = db.sql(q)
lines = ['# Terminal-label row inventory', '', 'For baseline fights with a forced random move, only decisions **after** the last random move have usable realised terminal labels. Self-play has no forced random move. Counts include all decision rows (not child rows).', '',
         '| ' + ' | '.join(rel.columns) + ' |', '| ' + ' | '.join(['---'] * len(rel.columns)) + ' |']
lines += ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rel.fetchall()]
(HERE / 'ROW_AUDIT.md').write_text('\n'.join(lines) + '\n')
print('Wrote ROW_AUDIT.md')
