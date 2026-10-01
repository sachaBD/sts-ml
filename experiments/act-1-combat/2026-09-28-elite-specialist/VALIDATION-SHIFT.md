# Validation: combat_v3/2026-09-27/elite-v2-validation-net vs guided-rollout MCTS (20k sims)

450 fixed bucket-5 fights (150 per elite); both replayed the identical original start. Positive HP-eq = net better. 95% intervals: 10,000 source-run-seed cluster bootstraps, percentile. This is model-selection data, **not the final test**.

| Encounter | Fights | HP-eq net − MCTS [95% CI] | Wins MCTS / net | Deaths MCTS / net | Net-only / MCTS-only wins |
|---|---:|---:|---:|---:|---:|
| gremlin_nob | 150 | +0.92 [-0.26, +2.20] | 142 / 143 | 8 / 7 | 2 / 1 |
| lagavulin | 150 | -1.18 [-3.33, +0.93] | 137 / 136 | 13 / 14 | 3 / 4 |
| three_sentries | 150 | -2.63 [-4.85, -0.44] | 124 / 120 | 26 / 30 | 3 / 7 |
| all | 450 | -0.96 [-2.07, +0.11] | 403 / 399 | 47 / 51 | 8 / 12 |
