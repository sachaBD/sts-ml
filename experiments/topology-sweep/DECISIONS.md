# Topology sweep decision log (Opus)
Compute budget: ≤ 8h from 08:00 BST 2026-09-30 (hard stop 16:00); aim ≈ 5h.

## Block 0 (07:10-08:12)
- Added: topology specs + frozen registry (topology/combat_outcome/), kind combat_outcome_v2, train_marginals `--topology` / `--arms`
  (default behaviour unchanged), eval_marginals.py (checkpoint re-scoring; verified max |Δp| 1e-7 vs training predictions).
- Smoke (s1 data, 2-4 epochs): v1-w64 and v2-w64 train fine; bigger nets overfit fast at lr .003 on small data -> lr {.003,.001} in grid.
- Launched 08:05: grid1 (36 runs, 2 parallel) ‖ hs16 generation (8 workers). First legacy spec run reproduces card-outcomes
  s3c-model-supp seed0 exactly. Measured: ~7 min/run under contention; hs16 slower than planned (~370 hard/elite fights/min).
  Expected block 1 length ~2.5-3h (to ~11:00). Accepted rather than shrinking hs16 (boss effects need it).

## Block 1 result (10:35-11:05)
- hs16: hard/elite 28800 fights/37 min, boss 7200/111 min (2969 deaths). co2 grid runs failed on an uppercase run id ("L2");
  fixed train_one.sh to lower-case ids, reran (logs of failed attempt in logs/failed-uppercase-id/). Smoke lines removed from FROZEN.tsv.
- hs16 confirms data growth helps card effects (legacy, 3 seeds): boss dMSE +0.30 (s1) -> -1.56 -> -3.69 (s3) -> -2.51 (s3c);
  elite -0.12 -> -1.09 -> -1.33 -> -1.64. Natural-only ~0. True pair-effect sd (hs16): boss 4.4 HP, elite 2.9, hard 1.3.
- v1 capacity: bigger nets early-stop at epoch 1-2 (overfit within ~65k samples); modest outcome gains (natural boss Brier
  .162 -> .148-.151), card effects not clearly better. Hypothesis: few distinct donor decks (many rows per donor) -> memorisation.
- co2-w32 lr .001 looks best on supp dev (boss dMSE -3.3 vs legacy -0.35; synthetic elite Brier .042 vs .048). Pending hs16.
## Block 2 (11:03): grid2 = 15 runs around co2 (lr .0003, depth 1/3, w64 at .0003) + v1-w32 lr .0003 control; 3 parallel.

## Grid 1 review (11:05; table diag/grid1.md, 3 seeds each)
- Outcomes: capacity helps boss outcome a little (natural boss Brier legacy .162 -> .148-.152 for w32/w64 at lr .001); synthetic boss ~.18 flat.
- Card effects on hs16 (precise pair means): co2 (encounter-conditioned cards) is the clearest gain on elites (R2 .15-.17 vs legacy .09; dMSE -2.2/-2.3 vs -1.6)
  and hard (-0.29/-0.40 vs -0.15). Boss card effects: nobody beats legacy lr .001/co1-w32 (hs dMSE ~ -3.1/-3.5); R2 vs true effects ~0 for all.
- Bigger nets early-stop at epoch 1-2 -> overfitting, consistent with few distinct donor decks.
## Block 3 (11:08): paired auxiliary loss (new opt-in CARD_OUTCOME_PAIR_W; squared error of predicted vs observed score difference
  vs base deck on the same group/encounter/fight seed). grid3 = {co2-w32-l2 lr .001, co1-w32 lr .001} x pair_w {1,4} x 3 seeds,
  running 1-parallel beside grid2 (3-parallel).

## Grid 2+3 review (12:20; diag/table-{tsweep,tpair1,tpair4}.md)
- Best: co2-w32-h64-l1-d30 lr .001 (1 residual block, dropout .3): natural boss Brier .142 [.139-.147] vs legacy .162; synthetic boss .169;
  hs16 elite card R2 .32 [.28-.41] vs legacy .09 [.06-.15]; hs boss dMSE -5.0 vs -2.5. Lower lr (.0003) also helped co2-l2 (.154 -> .247 elite R2).
  Pattern: less capacity/slower fitting generalises better -> overfitting to few donor decks is the binding constraint.
- Pair loss w=1: improves card effects (co2-l2 .001: elite R2 .15 -> .25, boss dMSE -2.6 -> -5.8) at small outcome cost; w=4 hurts outcomes.
## Grid 4 (12:22): does it stack? co2-w32-l1-d30 x {lr .0003} and x pair_w 1 x {lr .001, .0003}; 9 runs, then hs16 eval + REPORT.

## Close (12:35)
- Grid 4: pair loss stacks with co2-l1 (hs R2 boss .08 -> .21, elite .32 -> .35, hard .10 -> .28) at small outcome cost; lr .0003 no help for l1.
- Stopped at ~4.5h of 8h: remaining questions (confirmation, donor diversity) need fresh data/cohorts rather than more dev-set tuning. REPORT.md written.
