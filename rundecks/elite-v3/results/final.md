# Final test: elite-v3-t2 + policy priors (c_puct 1.0) vs MCTS on the 1,500 reserved fights

Baseline: `elite-v3-f-mcts`. Positive HP-eq = candidate better. 95% intervals: 10,000 source-run-seed cluster bootstraps (percentile).

## `elite-v3-f-candidate` vs baseline

| Encounter | Fights | HP-eq cand − base [95% CI] | Wins base / cand | Cand-only / base-only wins | Potions kept on wins base / cand |
|---|---:|---:|---:|---:|---:|
| gremlin_nob | 462 | +1.00 [+0.31, +1.70] | 441 / 442 | 4 / 3 | 0.11 / 0.11 |
| lagavulin | 479 | +0.98 [-0.09, +2.03] | 445 / 445 | 9 / 9 | 0.07 / 0.09 |
| three_sentries | 469 | -0.72 [-1.85, +0.38] | 403 / 395 | 10 / 18 | 0.13 / 0.12 |
| all | 1410 | +0.42 [-0.16, +0.98] | 1289 / 1282 | 23 / 30 | 0.10 / 0.11 |

Unpaired fights (in one run only, e.g. diverged replays): baseline-only 0, candidate-only 0.

