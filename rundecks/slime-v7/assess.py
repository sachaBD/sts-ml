"""Collate completed Stage A runs; no gameplay or training. Run from repo root."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest

HERE = Path('rundecks/slime-v7')
ROOT = Path('runs/schema=combat_v3/date=2026-09-24')
ARMS = {'r00': (15000, 8), 'r10': (60000, 8), 'r01': (15000, 32), 'r11': (60000, 32)}
frames, summaries, hashes = {}, {}, set()
for arm, (budget, particles) in ARMS.items():
    folder = ROOT / f'id=slime-v7-{arm}'
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'done'
    summary = json.loads((folder / 'out/summary.json').read_text())
    settings = summary['teacher']
    assert settings['simulations'] == budget and settings['particles'] == particles
    assert not settings['oracle'] and not settings['random_move']
    hashes.add(summary['worker']['sha256'])
    fights = {}
    for path in sorted((folder / 'out').glob('*.parquet')):
        columns = ['episode_id', 'row_kind', 'decision_index', 'starting_hp', 'starting_max_hp',
                   'encounter', 'won', 'final_hp', 'potions', 'terminal_value', 'simulations_used']
        for row in pq.ParquetFile(path).read(columns=columns).to_pylist():
            if row['row_kind'] != 'decision':
                continue
            fight = fights.setdefault(row['episode_id'], {'start': [row[k] for k in
                ('encounter', 'starting_hp', 'starting_max_hp')], 'seen': set(), 'simulations': 0,
                **{k: row[k] for k in ('won', 'final_hp', 'potions', 'terminal_value')}})
            assert row['decision_index'] not in fight['seen'], 'duplicate decision'
            fight['seen'].add(row['decision_index'])
            fight['simulations'] += row['simulations_used']
    assert len(fights) == summary['fights'] == 156
    assert set(fights) == set(summary['episodes'])
    frames[arm] = fights
    summaries[arm] = {'simulations': budget, 'particles': particles, 'fights': len(fights),
        'wins': sum(f['won'] for f in fights.values()),
        'terminal_value': float(np.mean([f['terminal_value'] for f in fights.values()])),
        'mean_final_hp_all_fights': float(np.mean([f['final_hp'] for f in fights.values()])),
        'mean_remaining_potions': float(np.mean([f['potions'] for f in fights.values()])),
        'seconds_per_fight': summary['seconds_per_fight'],
        'total_simulations': sum(f['simulations'] for f in fights.values()),
        'total_decisions': sum(len(f['seen']) for f in fights.values())}
    summaries[arm]['simulations_per_decision'] = summaries[arm]['total_simulations'] / summaries[arm]['total_decisions']
assert len(hashes) == 1
ids = sorted(frames['r00'])
for fights in frames.values():
    assert set(fights) == set(ids)
    assert all(fights[i]['start'] == frames['r00'][i]['start'] for i in ids)
values = {a: np.array([frames[a][i]['terminal_value'] for i in ids]) for a in ARMS}
# Resample whole paired fights; fixed seed, 20k percentile bootstrap replicates.
indices = np.random.default_rng(0).integers(0, len(ids), size=(20000, len(ids)))
def estimate(d):
    low, high = np.quantile(d[indices].mean(axis=1), [.025, .975])
    return {'diff': float(d.mean()), 'ci_low': float(low), 'ci_high': float(high)}
comparisons = {}
for base, cand in [('r00', 'r10'), ('r00', 'r01'), ('r00', 'r11'), ('r01', 'r11'), ('r10', 'r11'), ('r10', 'r01')]:
    d = values[cand] - values[base]
    gained = sum(frames[cand][i]['won'] and not frames[base][i]['won'] for i in ids)
    lost = sum(frames[base][i]['won'] and not frames[cand][i]['won'] for i in ids)
    comparisons[f'{cand}-vs-{base}'] = {**estimate(d), 'wins_gained': gained, 'wins_lost': lost,
        'mcnemar_p': binomtest(gained, gained + lost).pvalue if gained + lost else 1.0,
        'better_equal_worse': [int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())]}
interaction = estimate(values['r11'] - values['r01'] - values['r10'] + values['r00'])
result = {'n': len(ids), 'bootstrap_replicates': 20000, 'seed': 0, 'worker_sha256': next(iter(hashes)),
          'arms': summaries, 'comparisons': comparisons, 'interaction': interaction}
(HERE / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
