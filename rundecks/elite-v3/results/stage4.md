# Stage 4: t3 (expert-iteration fine-tune of t2) + policy priors, c_puct 1.0 (450 validation fights)

Baseline: `elite-v3-v-mcts`. Positive HP-eq = candidate better. 95% intervals: 10,000 source-run-seed cluster bootstraps (percentile).

## `elite-v3-v-t3-policy` vs baseline

| Encounter | Fights | HP-eq cand − base [95% CI] | Wins base / cand | Cand-only / base-only wins | Potions kept on wins base / cand |
|---|---:|---:|---:|---:|---:|
| gremlin_nob | 142 | +1.42 [-0.01, +2.94] | 134 / 134 | 2 / 2 | 0.09 / 0.16 |
| lagavulin | 137 | +0.57 [-1.60, +2.74] | 125 / 125 | 4 / 4 | 0.08 / 0.10 |
| three_sentries | 142 | -1.35 [-3.66, +0.80] | 117 / 111 | 2 / 8 | 0.15 / 0.21 |
| all | 421 | +0.21 [-0.96, +1.39] | 376 / 370 | 8 / 14 | 0.10 / 0.16 |

Unpaired fights (in one run only, e.g. diverged replays): baseline-only 0, candidate-only 0.

