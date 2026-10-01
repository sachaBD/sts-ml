#!/usr/bin/env python3
"""Paired validation scores; cluster-bootstrap by source run seed (95% percentile CI)."""
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
from sts_combat_rl.query import connect

HERE = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument('--candidate', default='combat_v3/2026-09-27/elite-v1-validation-net')
p.add_argument('--output', default='VALIDATION.md')
a = p.parse_args()
A = 'combat_v3/2026-09-27/elite-v1-validation-mcts'
B = a.candidate
db = connect()
q = f"""
select a.episode_id, a.run_seed, a.encounter, a.starting_max_hp,
       a.won m_won, b.won n_won, a.terminal_value m_value, b.terminal_value n_value,
       a.starting_hp m_hp, b.starting_hp n_hp
from (select * from combat_v3 where run_id = '{A}' and row_kind='decision' and decision_index=0) a
join (select * from combat_v3 where run_id = '{B}' and row_kind='decision' and decision_index=0) b
  on a.episode_id=b.episode_id
where a.run_seed=b.run_seed and a.encounter=b.encounter and a.starting_hp=b.starting_hp
      and a.starting_max_hp=b.starting_max_hp
"""
rows=db.sql(q).fetchall()
assert len(rows)==450, f'expected all 450 exact-start pairs, got {len(rows)}'
assert len(set(r[0] for r in rows))==450
rng=np.random.default_rng(2026)
lines=[f'# Validation: {B} vs guided-rollout MCTS (20k sims)', '',
       '450 fixed bucket-5 fights (150 per elite); both replayed the identical original start. Positive HP-eq = net better. 95% intervals: 10,000 source-run-seed cluster bootstraps, percentile. This is model-selection data, **not the final test**.', '',
       '| Encounter | Fights | HP-eq net − MCTS [95% CI] | Wins MCTS / net | Deaths MCTS / net | Net-only / MCTS-only wins |',
       '|---|---:|---:|---:|---:|---:|']
for enc in ('gremlin_nob','lagavulin','three_sentries','all'):
    v=[r for r in rows if enc=='all' or r[2]==enc]
    diffs=np.array([(r[7]-r[6])*(55+r[3]) for r in v])
    groups=defaultdict(list)
    for r,d in zip(v,diffs): groups[r[1]].append(d)
    groups=list(groups.values())
    sums=np.array([sum(g) for g in groups]); counts=np.array([len(g) for g in groups]); k=len(groups)
    idx=rng.integers(0,k,(10000,k))
    boots=sums[idx].sum(axis=1)/counts[idx].sum(axis=1)
    lo,hi=np.quantile(boots,[.025,.975])
    wm=sum(r[4] for r in v); wn=sum(r[5] for r in v)
    only_n=sum(r[5] and not r[4] for r in v); only_m=sum(r[4] and not r[5] for r in v)
    lines.append(f'| {enc} | {len(v)} | {diffs.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {wm} / {wn} | {len(v)-wm} / {len(v)-wn} | {only_n} / {only_m} |')
(HERE/a.output).write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
