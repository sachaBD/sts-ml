# Outcome-model paired gauntlet pilot (2026-09-29)

- Gauntlet run: `gauntlet_v1/2026-09-29/outcome-pilot` (config `apps/gauntlet/outcome-pilot.toml`; 20 reward states from
  dev buckets 0-1 of `combat_v3/2026-09-27/act1-eval-mcts-a20`, 8 shared seeds, 3 elites at the nearest elite distance
  + boss; guided_rollout, 8 particles, elite 5000 / boss 15000; 2368 fights, 15.8 min on 10 workers).
- Model: frozen `combat_outcome_v1/2026-09-29/act1-eval-mcts-v1`.
- Scoring: `PYTHONPATH=python .venv/bin/python apps/combat_transition/score_gauntlet.py gauntlet_v1/2026-09-29/outcome-pilot
  combat_outcome_v1/2026-09-29/act1-eval-mcts-v1 --out experiments/act-1-card-selection/outcome-model-pilot`
  -> `score_gauntlet.json` (paired option-vs-SimpleAgent deltas by kind, levels by encounter, distribution shift),
  `pairs.json`.
- Descriptive pilot. Bootstrap intervals resample source run seeds only.

## Status

Development set: dev buckets 0-1 were used to pick and inspect this pilot, so it is not a future untouched confirmation.
Descriptive results are in `score_gauntlet.json`.
- Boss: win and HP levels are heavily over-predicted (e.g. Hexaghost real win 0.06 vs predicted 0.50). Paired deltas
  are worse than the card-invariant (delta 0) reference.
- Elite: aggregate win rate per encounter is close to observed. That is an encounter average; state-level calibration
  is not tested. Ending HP is over-predicted by 5-8. Paired HP / score deltas correlate about 0.6 with real deltas
  (54 pairs). The MAE is not reliably better than delta 0.
- Training-support gap (pre state, ok fights of the training cohort vs pilot fights). Share of fights with <= 1 upgraded card
  and <= 2 relics: training boss 0.1% (1/722), pilot boss 70%; training elite 38%, pilot elite 78%.
  Pilot `summary.json` `teacher.simulations` shows only the last result's value. Per-row `simulations` (elite 5000 /
  boss 15000) is authoritative.
