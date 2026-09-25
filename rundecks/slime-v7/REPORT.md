# slime-v7 Stage A: search-budget / particle assessment

**Conclusion:** Increasing guided-rollout search from 15k to 60k simulations did not improve observed play at either particle count. Increasing particles from 8 to 32 produced a modest, directionally consistent improvement, but uncertainty includes no effect. **15k / 32 is the most promising development candidate, not a confirmed stronger expert.**

## Setup and checks

- Same 156 v6 gen0 validation fights, now treated as a development benchmark.
- Guided-rollout leaves, public information only, mean backup, random move off, 10 workers. Four arms executed sequentially.
- Simulation numbers are caps. Early stopping, forced-action budget, rollout policy, and other search controls unchanged.
- All four runs completed, with the same worker binary hash. Episode sets and recorded encounter/starting HP/max HP match across arms.
- R00 reproduced all 4,119 v6 baseline decision actions and their recorded outcomes before the batch proceeded.
- No training or neural/hybrid evaluation occurred in Stage A.

## Main results

Terminal-value intervals are paired percentile bootstrap 95% CIs over 156 fights (20,000 resamples, seed 0). Exploratory comparisons, no multiplicity correction. Wins gained/lost are relative to R00.

| Arm | Simulation cap | Particles | Wins / 156 | Terminal value | Difference vs R00 (95% CI) | Wins gained / lost | Seconds/fight |
|---|---:|---:|---:|---:|---|---:|---:|
| R00 | 15k | 8 | 121 | 0.4274 | baseline | — | 9.7 |
| R10 | 60k | 8 | 118 | 0.4188 | −0.0086 (−0.0336, +0.0152) | 7 / 10 | 32.9 |
| **R01** | **15k** | **32** | **126** | **0.4412** | **+0.0138 (−0.0122, +0.0404)** | **13 / 8** | **8.8** |
| R11 | 60k | 32 | 125 | 0.4400 | +0.0126 (−0.0090, +0.0348) | 9 / 5 | 36.1 |

Exact McNemar p-values versus R00: R10 0.629, R01 0.383, R11 0.424. Neither wins nor terminal value establishes a clear improvement over baseline.

### Factor comparisons

| Change | Terminal-value difference (95% CI) |
|---|---|
| 15k → 60k at 8 particles | −0.0086 (−0.0336, +0.0152) |
| 15k → 60k at 32 particles | −0.0012 (−0.0234, +0.0212) |
| 8 → 32 particles at 15k | +0.0138 (−0.0122, +0.0404) |
| 8 → 32 particles at 60k | +0.0213 (−0.0046, +0.0481) |

Interaction, `(R11−R01)−(R10−R00)`: +0.0074 (−0.0268, +0.0428). No clear evidence that the larger simulation budget unlocks a benefit from more particles. These comparisons share fights and are not independent replications.

### Actual compute

| Arm | Total simulations | Decisions | Simulations/decision | Reported gameplay wall time |
|---|---:|---:|---:|---:|
| R00 | 43,301,000 | 4,119 | 10,513 | 2.6 min |
| R10 | 147,209,000 | 3,948 | 37,287 | 9.0 min |
| R01 | 44,448,500 | 4,110 | 10,815 | 2.4 min |
| R11 | 162,142,500 | 4,113 | 39,422 | 9.8 min |

The 60k arms used 3.40× and 3.65× the actual simulations of their corresponding 15k arms, respectively. Runtime rose about 3.38× and 4.08×. Thus the null performance result is not because early stopping prevented additional search work.

Seconds/fight include worker/replay overhead and reflect concurrent execution. Do not interpret R01's slightly lower time than R00 as a demonstrated speedup from more particles. Host wall-clock timestamps jump during the batch; use the reported monotonic durations, not timestamp subtraction, for runtime interpretation. Summed reported gameplay wall time was about 24 minutes, excluding launch/build overhead.

## Assessment

1. **More search alone is not a productive next lever on this evidence.** Both particle counts show flat/slightly lower scores at 60k for substantially greater compute. This is not proof of a hard search ceiling, but does not justify a larger simulation sweep now.
2. **Broader uncertainty sampling remains plausible.** Both budgets favor 32 particles numerically. R01 is cheap enough to carry forward as a candidate setting, but its +5 net wins include 13 rescued fights and 8 new losses: it is not uniformly better.
3. **We have not yet found a demonstrably stronger expert to distill.** Do not launch a training generation on the assumption that R01 labels are definitively better than R00 labels.
4. **The data do not diagnose rollout bias or prove outcome-only learning will help.** Those remain hypotheses. Particle approximation, evaluator quality, search structure, and model supervision are still possible limits. The result also does not quantify recoverable oracle headroom.

## Recommended next decision

- Stop increasing guided-rollout simulation budgets for now. Keep R00 as the historical reference and R01 as the provisional best search setting.
- If continuing the planned search investigation, use the 15k / 32 setting for Stage B: frozen gen0 immediate leaves versus one predefined bounded-rollout/gen0 hybrid versus the completed R01 rollout reference. Preserve the original gen0 reference for context. This tests evaluator quality without confounding it with retraining.
- If prioritizing learning instead, design the single-generation terminal-score-only pilot described in `../../slop_docs/slime/terminal_score_iteration.md`. Stage A motivates looking beyond compute, but does not establish this learning approach as the solution.
- Before calling a selected expert stronger, confirm on fresh fights. The 156-fight development set has already been used for repeated selection.

No further gameplay runs launched as part of this assessment.

## Reproduction and artifacts

- Runs: `combat_v3/2026-09-24/slime-v7-r00`, `-r10`, `-r01`, `-r11`.
- Configs: `configs/slime-v7-r{00,10,01,11}.toml`.
- Batch log: `logs/stage-a.log`; each run also has registered stdout/stderr logs.
- Numeric results: `assessment.json`.
- Recompute collation: `.venv/bin/python rundecks/slime-v7/assess.py` from repository root. Reads compacted or un-compacted Parquet, checks completed runs/settings/cohort consistency, and resamples paired fights.
- Worker SHA256: `8994fcb01774671515031d1b755e8db781a78aabc90bf7cab5aed73d3a7fd89b`.
