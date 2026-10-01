# Validation: elite-terminal-v1 vs guided-rollout MCTS (20k sims)

450 fixed bucket-5 fights (150 per elite); both replayed the identical original start. Positive HP-eq = net better. 95% intervals: 10,000 source-run-seed cluster bootstraps, percentile. This is model-selection data, **not the final test**.

| Encounter | Fights | HP-eq net − MCTS [95% CI] | Wins MCTS / net | Deaths MCTS / net | Net-only / MCTS-only wins |
|---|---:|---:|---:|---:|---:|
| gremlin_nob | 150 | +1.19 [-0.08, +2.61] | 142 / 145 | 8 / 5 | 3 / 0 |
| lagavulin | 150 | -0.82 [-3.09, +1.30] | 137 / 133 | 13 / 17 | 2 / 6 |
| three_sentries | 150 | -4.21 [-6.65, -1.83] | 124 / 116 | 26 / 34 | 2 / 10 |
| all | 450 | -1.28 [-2.50, -0.09] | 403 / 394 | 47 / 56 | 7 / 16 |
