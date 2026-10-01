# Outcome-model topology sweep: report (Opus, 2026-09-30 12:35 BST)

Compute used: about 4.5 h of the 8 h allowed (08:05–12:30). Runs: 72 training runs (36 + 15 + 12 + 9) plus one high-seed evaluation set (hs16).

## Summary
- **New default candidate: `co2-w32-h64-l1-d30`** (encounter-conditioned card encoder, sum+mean pooling, one residual
  block), trained at lr .001 **with the paired auxiliary loss, weight 1** (`CARD_OUTCOME_PAIR_W=1`). Compared with the
  legacy `co1-w16-h32-d30` (3 training seeds each):
  - Elite card-effect R² on hs16 rises from .09 to .35. The seed ranges do not overlap.
  - Boss R² rises from −.05 to .21, and hard from −.07 to .28.
  - Outcome prediction also improves: natural boss Brier .162 → .144 and synthetic elite Brier .048 → .042.
- **Without the pair loss, the same topology is the best outcome model:** natural boss Brier .142 and synthetic boss .169.
  Its card effects are good on elites (R² .32) but weaker on bosses (.08).
- **Raw capacity does not help.** Wider or deeper nets early-stop at epoch 1–2 and do not improve card effects. What
  helped was a less aggressive fit: fewer residual blocks, more dropout, and a lower learning rate. The binding constraint
  looks like **overfitting to a small pool of distinct donor decks**, not model size. That points to more natural donors
  (or more varied generated decks) as the next data lever.
- **Card effects are still hard to predict.** Even the best model explains only about 20–35% of the real card-effect
  variance. That variance has a standard deviation of 4.4 HP for bosses, 2.9 for elites and 1.3 for hard fights, measured
  on hs16.

## Main results
Mean [min..max] over training seeds 0–2. The dev columns use the supplementary dev set: fixed eval plus devsupp. The hs16
columns use 16-seed pair means on bucket-8 donors, never trained on. R² is the explained fraction of the true
(noise-corrected) pair-effect variance; predicting zero scores about −.1 to −.2. Full tables: `diag/table-{tsweep,tpair1,tpair4}.md`.

| Model | Brier natural boss | Brier synthetic boss | Brier synthetic elite | hs R² boss | hs R² elite | hs R² hard |
|---|---|---|---|---|---|---|
| Legacy co1-w16-h32-d30, lr .003 | .162 [.159–.168] | .180 [.171–.187] | .048 [.046–.052] | −.05 [−.14–.03] | .09 [.06–.15] | −.07 [−.20–.03] |
| co1-w32-h64-d30, lr .0003 (best v1) | .148 | .177 | .046 | .01 | .10 | – |
| co2-w32-h64-l1-d30, lr .001 | **.142** [.139–.147] | **.169** [.165–.172] | .043 [.041–.045] | .08 [−.02–.21] | .32 [.28–.41] | .10 [−.04–.18] |
| **co2-w32-h64-l1-d30, lr .001 + pair loss w1** | .144 [.131–.166] | .175 [.159–.199] | **.042** [.040–.044] | **.21** [.09–.29] | **.35** [.25–.44] | **.28** [.14–.38] |
| co2-w32-h64-l2-d20, lr .001 (+pair w1) | .152 (.149) | .178 (.183) | .046 (.044) | −.05 (.12) | .15 (.25) | – |
| co2-w64-h128-l2-d20, lr .0003 | .139 | .165 | .044 | −.07 | .21 | – |
| Pair loss w4 (co2-l2 / co1-w32) | .178 / .192 | .208 / .221 | .046 / .047 | .08 / −.03 | .23 / .09 | – |

**Other runs:**
- The lr .003 runs were worse for every topology larger than the legacy one.
- `co2-l3` (3 residual blocks) was no better than `co2-l2`.
- Lowering the lr from .001 to .0003 helped `co2-l2` but did not help `co2-l1`, which is already regularised.

**hs16 data-growth check:** the card-outcomes checkpoints s1 → s3c use the legacy topology. On hs16, boss card dMSE vs zero
goes +0.3 → −1.6 → −3.7 → −2.5, and elite −0.1 → −1.1 → −1.3 → −1.6. This confirms that the synthetic data drives the
card-effect learning.

## Caveats
- **Tuning bias.** hs16 and the supplementary dev set were inspected repeatedly while choosing configurations. The winner
  was picked from about 27 configurations, so expect some regression on fresh data. The recommendation needs confirmation
  on a new held-out cohort; natural buckets 2–3 are still reserved.
- **Pair-loss variance.** The pair-loss winner has a wider outcome-Brier spread across seeds: natural boss .131–.166.
- **Recipe change.** The pair loss changes the training objective. The stopping criterion is still the joint NLL.
- **Scope.** hs16 covers bosses, elites and hard fights (bucket-8 donors) only; easy fights are not included.
- **Card sampling.** Card effects are measured for generator-sampled candidate cards. These may not match real reward
  distributions.
- **Not tested.** Nothing here tests gameplay.

## What was built
- **`topology/combat_outcome/`:** named specs (kind + args) and `FROZEN.tsv`, which `train_marginals.py` appends to and
  enforces. All 8 specs are now frozen, and so is the new kind `combat_outcome_v2`
  (`python/sts_combat_rl/topology/combat_outcome_v2.py`).
- **`apps/combat_transition/train_marginals.py`:** new opt-in `--topology`, `--arms` and `CARD_OUTCOME_{LR,SEED,PAIR_W}`
  settings. The defaults reproduce the earlier runs exactly.
- **`apps/combat_transition/eval_marginals.py`:** re-scores saved checkpoints on other data. Verified to within 1e-7 of
  the training-time predictions.
- **`experiments/topology-sweep/`:** RUNBOOK, `sweep.sh` / `train_one.sh` (resumable, lower-cased ids), grids, `DECISIONS.md`,
  `diag/table.py`.
- **hs16 data:** `card_marginals_v1/2026-09-30/card-outcomes-hs16-{hard-elite,boss}`.
- **Incident:** the first co2 attempt failed on an uppercase run id. It was fixed and rerun, and the logs are kept.

## Recommended next steps
1. **Confirm the winner on fresh data.** Train `co2-w32-h64-l1-d30` with pair loss w1 using 5 seeds and evaluate once on
   a fresh, untouched cohort: a new hs16-style set on unused donors, or the reserved buckets.
2. **Address the donor bottleneck.** Generate more natural donor runs or diversify donor decks; the evidence points to
   this over a larger model. A cheap test is a learning curve over the number of distinct donors.
3. **If a gameplay test follows,** use the pair-loss model to rank card picks against the current picker in an A/B game
   evaluation.
