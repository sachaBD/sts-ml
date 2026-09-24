# slime-v6 report

Test set: gen0's 156 held-out Slime fights. Random move off. Terminal value is the mean terminal_value. Diffs are paired, with 95% CI.

| model | wins / 156 | terminal value | vs teacher | vs previous gen |
|---|---|---|---|---|
| teacher (guided rollout) | 121 | 0.427 | – | – |
| **gen0** | **115** | **0.401** | −0.027 (−0.055, +0.002) | – |
| gen1 (w 0.25), chosen | 114 | 0.400 | −0.027 (−0.055, +0.001) | −0.001 (−0.022, +0.021) |
| gen1 (w 0.5) | 112 | 0.399 | −0.029 (−0.058, −0.000) | −0.002 (−0.028, +0.023) vs gen0 |
| gen2 | 110 | 0.393 | −0.035 (−0.064, −0.005) | −0.007 (−0.023, +0.007) |
| gen3 | 106 | 0.382 | −0.045 (−0.076, −0.015), p = 0.018 | −0.011 (−0.030, +0.008) |

gen3 vs gen0 directly: −0.019 (−0.042, +0.003); wins 106 vs 115 (McNemar p = 0.064).

| DAgger round | learner | learner wins / 625 | teacher disagreement | disagreement cost: median / share > 0.02 |
|---|---|---|---|---|
| 1 | gen0 | 441 | 33.4% | 0.0019 / 15.5% |
| 2 | gen1-w25 | 438 | 32.7% | 0.0019 / 16.7% |
| 3 | gen2 | 441 | 31.9% | 0.0018 / 16.3% |

Disagreement cost is the teacher's own estimate: the mean value of its most-visited move minus that of the learner's move.

**Conclusion.** DAgger did not help. **gen0 is the best model.**
- gen0 plays about 0.03 terminal value (roughly 2 HP) below the teacher, which is not significant.
- Each DAgger generation was flat or slightly worse. Over three rounds the decline adds up to a significant gap to the teacher by gen3.
- Most disagreements with the teacher are near-ties by the teacher's own estimate.
- The costly ~16% did not shrink across rounds. The corrections (teacher root_value on learner states) don't change the moves that matter; if anything they add noise.
- Not tested: whether more bootstrap data improves gen0.

## Runs (all 2026-09-24 unless noted)

| step | run id |
|---|---|
| bootstrap | combat_v3/…/act1-a20-7, plus act1-a20, -1, -2, -5, -6 (2026-09-23): 781 Slime fights |
| gen0 | value_net_v1/…/slime-v6-gen0 (625 train / 156 test fights) |
| teacher play | combat_v3/…/slime-v6-teacher-play |
| DAgger | combat_v3/…/slime-v6-dagger1, -dagger2, -dagger3 |
| models | value_net_v1/…/slime-v6-gen1-w25, -gen1-w50, -gen2, -gen3 |
| plays | combat_v3/…/slime-v6-{gen0,gen1-w25,gen1-w50,gen2,gen3}-play |
| compares | fight_comparison_v1/…/slime-v6-*-vs-* |

Configs: `rundecks/slime-v6/configs/`. Timeline: `LOG.md`.
