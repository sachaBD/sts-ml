# Stage 3a: v3 arms vs MCTS (450 validation fights)

Baseline: `elite-v3-v-mcts`. Positive HP-eq = candidate better. 95% intervals: 10,000 source-run-seed cluster bootstraps (percentile).

## `elite-v3-v-t1-value` vs baseline

| Encounter | Fights | HP-eq cand − base [95% CI] | Wins base / cand | Cand-only / base-only wins | Potions kept on wins base / cand |
|---|---:|---:|---:|---:|---:|
| gremlin_nob | 142 | +0.08 [-1.52, +1.72] | 134 / 133 | 2 / 3 | 0.09 / 0.12 |
| lagavulin | 137 | -2.24 [-4.75, +0.13] | 125 / 121 | 3 / 7 | 0.08 / 0.11 |
| three_sentries | 142 | -2.24 [-4.99, +0.39] | 117 / 111 | 5 / 11 | 0.15 / 0.15 |
| all | 421 | -1.46 [-2.78, -0.11] | 376 / 365 | 10 / 21 | 0.10 / 0.13 |

Unpaired fights (in one run only, e.g. diverged replays): baseline-only 0, candidate-only 0.

## `elite-v3-v-t1-policy` vs baseline

| Encounter | Fights | HP-eq cand − base [95% CI] | Wins base / cand | Cand-only / base-only wins | Potions kept on wins base / cand |
|---|---:|---:|---:|---:|---:|
| gremlin_nob | 142 | +1.03 [+0.02, +2.18] | 134 / 135 | 1 / 0 | 0.09 / 0.13 |
| lagavulin | 137 | +0.32 [-2.03, +2.69] | 125 / 124 | 4 / 5 | 0.08 / 0.11 |
| three_sentries | 142 | -0.53 [-2.73, +1.64] | 117 / 114 | 4 / 7 | 0.15 / 0.16 |
| all | 421 | +0.27 [-0.85, +1.42] | 376 / 373 | 9 / 12 | 0.10 / 0.13 |

Unpaired fights (in one run only, e.g. diverged replays): baseline-only 0, candidate-only 0.

## `elite-v3-v-t2-policy` vs baseline

| Encounter | Fights | HP-eq cand − base [95% CI] | Wins base / cand | Cand-only / base-only wins | Potions kept on wins base / cand |
|---|---:|---:|---:|---:|---:|
| gremlin_nob | 142 | +1.50 [+0.04, +3.08] | 134 / 136 | 3 / 1 | 0.09 / 0.13 |
| lagavulin | 137 | -0.01 [-2.45, +2.40] | 125 / 124 | 5 / 6 | 0.08 / 0.09 |
| three_sentries | 142 | +0.94 [-1.27, +3.15] | 117 / 118 | 6 / 5 | 0.15 / 0.10 |
| all | 421 | +0.82 [-0.39, +2.07] | 376 / 378 | 14 / 12 | 0.10 / 0.11 |

Unpaired fights (in one run only, e.g. diverged replays): baseline-only 0, candidate-only 0.

