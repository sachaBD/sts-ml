# Iteration 1: promising outcome fit, card effects untested

Artifacts:
- `runs/schema=combat_transition_v1/date=2026-09-29/id=act1-eval-mcts-a20/out/`
- `runs/schema=combat_outcome_v1/date=2026-09-29/id=act1-eval-mcts-v1/out/report.md` (read directly), `report.json`, `model.pt`

## What ran

One final training seed. Small residual Deep Sets model, 19,850 parameters, initialised around an encounter/start-HP baseline. Survival plus conditional battle-ending HP histogram (5-HP bins, explicit >100 tail). Post-exit transitions not modelled. Three configurations explored; earlier dev results were seen. Treat all results as development evidence, not untouched confirmation.

Training: 3,165 fights / 500 source runs (buckets 4–8); early stopping: 592 / 100 (bucket 9). Development: 1,238 / 200 (0–1), 74 deaths. Buckets 2–3 excluded. Source policy and extraction described in combat-model.md.

## Results

Model minus baseline; 95% paired run-seed cluster-bootstrap intervals, 1,000 resamples. These intervals do not include model-selection or training-seed uncertainty.

- Survivor HP MAE: 7.409 -> 6.526 HP; difference -0.883 [-1.124, -0.651], 1,164 won fights.
- Boss survivor HP MAE: 15.873 -> 11.762 HP; difference -4.111 [-5.773, -2.489], 91 won fights.
- Overall Brier difference: -0.005 [-0.008, -0.001], 1,238 fights.
- Boss log loss difference: +0.006 [-0.094, +0.131], 140 fights; no demonstrated improvement.
- Elite Brier difference: -0.010 [-0.018, -0.003], 228 fights.
- Easy/hard death counts in dev: 0 / 2. Cannot assess rare-death prediction reliably.
- Nominal 80% HP interval coverage: model 94.8% overall, but Slime Boss only 70% (30 wins). Aggregate coverage is not evidence of encounter-wise calibration; discrete-bin construction also matters.
- Saved report: CPU batch-1024 throughput 212,445 states/s. Single noisy measurement, competing workload; variability unknown.

## Data caveat

Implementation reports potion availability 15% before divergent fights versus 46% before successful replays. Thus exclusions are state-dependent. Exact battle outcome matches do not prove correct pre-combat reconstruction; opening-state comparison remains to be checked. No causal explanation for the potion association established.

## Next step

Bounded existing-data opening-state check, then a small paired-option gauntlet pilot with frozen model and matching combat budgets. No more architecture search or broad data expansion yet. The key question remains whether predicted **card-induced differences** agree with real combat outcomes. No card-selection improvement has been demonstrated.

## Follow-up: opening states and paired pilot

Implementation reports checked extraction `combat_transition_v1/2026-09-29/act1-eval-mcts-a20-checked`: all 6,217 usable fights match stored opening encounter/floor/HP/global/card/monster encoding; all 180 divergent fights also match at opening. This supports combat-visible input fidelity, not unencoded macro fields. Persistent extraction columns reportedly unchanged, so no retraining needed. Potion-associated exclusion bias remains; its mechanism is not established.

Approved paired pilot: 20 development reward states, 8 matched samples, nearest elite distance (three elite types) and known boss; frozen model, real combat budgets elite 5k/boss 15k. Estimated ~20 minutes / 10 workers from one smoke, variable under contention. Artifact target `gauntlet_v1/2026-09-29/outcome-pilot`. Report elites/bosses separately, augmentation counts/deck sizes, and paired card-effect errors against a zero-effect baseline. This is exploratory, not a policy-win claim.

One smoke sample had optimistic boss predictions and losses across all alternatives. It flags possible distribution shift, not established miscalibration: alternatives share RNG, and future augmentation must be inspected before calling all early-reward boss decks out of distribution. Score expectation must include deaths: startHP - p(win)*E[endHP|win] + penalty*(1-p(win)). Conditional survivor HP differences alone are insufficient.

## Paired pilot result: not yet useful for card selection

Completed `gauntlet_v1/2026-09-29/outcome-pilot`: 20 reward states / 19 source seeds, 2,368 fights, 8 matched samples, 15.8 minutes on 10 workers. Scoring artifacts and provenance: `experiments/act-1-card-selection/outcome-model-pilot/`. Results below reported by implementation; linked README inspected. No further jobs approved.

Paired score-difference MAE relative to predicting zero card effect:
- Elite: model 3.83 versus 4.34; difference -0.52, exploratory 95% source-seed bootstrap CI [-1.43, +0.35], 54 option pairs / 17 seeds. Correlation 0.60 is encouraging but does not establish useful choices.
- Boss: model 7.08 versus 5.35; difference +1.73 [+0.09, +3.53], 60 pairs / 19 seeds. Worse than zero-effect baseline.

Boss mean win probabilities were severely optimistic (real/model: Hexaghost .06/.50, Slime .39/.69, Guardian .30/.70). Elite mean win rates were closer, but unconditional ending HP was overpredicted by roughly 5–8 HP. Encounter-average agreement is not state-level calibration.

Measured shift: augmented boss deck size 16.9 versus training 16.7, but upgrades 1.15 versus 3.84, relics 1.95 versus 3.9, potions .35 versus .76. Gauntlet augmentation adds cards, not upgrades/relics/shops/rest effects. This supports a training-support concern, not proof that any one feature causes the error. Model feature use and support cannot be separated by this pilot.

Working recommendation for discussion: train on inputs representative of the intended gauntlet, rather than adjust the benchmark solely to flatter the model. Existing HP/RNG-only resamples will not supply missing deck/upgrade/relic diversity. Improving gauntlet realism is a separate design question. Elite-only deployment is not justified by the current evidence. Small, early-floor-heavy pilot and noisy observed card effects limit conclusions; training-seed and selection uncertainty are excluded from intervals.
