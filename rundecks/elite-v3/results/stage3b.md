# Stage 3b: c_puct sweep for best = elite-v3-t2 (450 validation fights)

Baseline: `elite-v3-v-mcts`. Positive HP-eq = candidate better. 95% intervals: 10,000 source-run-seed cluster bootstraps (percentile).

## `elite-v3-v-elite-v3-t2-c05` vs baseline

| Encounter | Fights | HP-eq cand − base [95% CI] | Wins base / cand | Cand-only / base-only wins | Potions kept on wins base / cand |
|---|---:|---:|---:|---:|---:|
| gremlin_nob | 142 | +1.41 [-0.08, +3.04] | 134 / 135 | 3 / 2 | 0.09 / 0.15 |
| lagavulin | 137 | +0.66 [-1.52, +2.85] | 125 / 127 | 5 / 3 | 0.08 / 0.08 |
| three_sentries | 142 | +0.39 [-1.85, +2.66] | 117 / 117 | 6 / 6 | 0.15 / 0.09 |
| all | 421 | +0.82 [-0.37, +2.03] | 376 / 379 | 14 / 11 | 0.10 / 0.11 |

Unpaired fights (in one run only, e.g. diverged replays): baseline-only 0, candidate-only 0.

## `elite-v3-v-elite-v3-t2-c2` vs baseline

| Encounter | Fights | HP-eq cand − base [95% CI] | Wins base / cand | Cand-only / base-only wins | Potions kept on wins base / cand |
|---|---:|---:|---:|---:|---:|
| gremlin_nob | 142 | +0.93 [-0.58, +2.57] | 134 / 135 | 3 / 2 | 0.09 / 0.13 |
| lagavulin | 137 | -0.50 [-2.72, +1.69] | 125 / 124 | 4 / 5 | 0.08 / 0.10 |
| three_sentries | 142 | -0.49 [-2.83, +1.81] | 117 / 113 | 4 / 8 | 0.15 / 0.15 |
| all | 421 | -0.01 [-1.20, +1.16] | 376 / 372 | 11 / 15 | 0.10 / 0.12 |

Unpaired fights (in one run only, e.g. diverged replays): baseline-only 0, candidate-only 0.

