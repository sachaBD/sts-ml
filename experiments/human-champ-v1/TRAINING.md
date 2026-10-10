# Follow-up: human-deck combat learning, round 1

Status: first real round completed successfully (32 minutes); candidate retained for
inspection but not promoted. No architecture/search-tree/objective changes.

Additional pull completed: 800 files in 718 seconds, 39,720 Ironclad A20 runs,
2,617 Champ fights, 1,133 usable reconstructed decks. No failed downloads/throttling.
Round 1 uses 800 training decks, 113 validation decks, and 100 old-regression decks
(two seeds); 220 eligible training-side decks remain unused in this bounded round.

## Hypothesis

D5's search improves when trained on realistic human-derived decks and learner-visited
states, with near-flat policy supervision reduced. This is a combined first feasibility
experiment, not an isolated causal test of policy weighting. Pure self-play versus teacher
mixing or weighting ablations can follow if the pipeline produces useful improvement.

## Actioned design

- Reuse run_rl app: `apps/run_rl/combat_loop.py`; no duplicate learner or native worker.
- Download up to 800 fresh 2020 files, four download workers, excluding prior done files.
- Keep supported, exact reconstructed decks. Exclude original benchmark IDs and families.
- New source-family 90/10 split; explicit manifests override legacy seed-hash splits.
- No gameplay augmentation. One human-seed training start per source deck. All original
  benchmark decks remain outside training; a fixed 100-deck subset is regression-evaluated.
- Initialize from D5, collect up to 800 learner-search trajectories at 2k simulations,
  with root noise/early-turn sampling; use all completed decision states.
- Add up to 128 MCTS20k full recovery/coverage trajectories (half ordinary coverage,
  then learner losses). Not per-state teacher annotation/DAgger yet.
- Outcome-only value labels. Policy loss scaled by normalized top-visit concentration;
  uniform policies have zero weight. No loss-state filtering or artificial sharpening.
- Train three epochs at lr=1e-4, gradient norm cap=1; frozen worker/model snapshots.
- Evaluate candidate and D5 on identical new-validation and regression starts, 2k sims.
  Paired coverage, failure/cap counts, win gap and family-clustered SE are reported.
- Conservative promotion; uncertain candidates do not replace D5. One real round first.

## Verification

16 focused unit tests passed (split isolation, missing-ID rejection, concentration/padding,
value-label preservation, teacher selection, and paired completion/SE accounting).
Pipeline smoke: native collect -> canonical replay -> teacher trajectories -> weighted
training -> ONNX export -> native candidate play -> comparison, completed successfully.
Budgets: 8 training decks, 11 validation decks, 4 regression decks (two seeds), learner
16 sims, teacher 64 sims, 1 CPU epoch. Smoke model is not a strength result or real-round
initialization; original D5 remains unchanged.

## Artifacts / progress

- Pull: `runs/schema=megacrit_runs_v1/date=2026-10-05/id=human-train-pull2020/`.
- Smoke: `runs/schema=combat_v4/date=2026-10-05/id=human-combat-loop-smoke/`.
- Real round: `runs/schema=combat_v4/date=2026-10-05/id=human-combat-r01/`.
- Controller usage/details: `apps/run_rl/COMBAT.md`.

## First-round result

792/800 learner trajectories completed (7 unsupported-state errors, 1 cap); all 128
teacher trajectories completed. Training used **44,510 decision states**: 37,701 learner
and 6,809 teacher. Three epochs; the existing held-out-loss criterion selected epoch 1.
Uniform/near-flat labels were downweighted; effective policy-state mass was about 12,462
out of 38,019 policy states. This does not establish that weighting caused the improvement.

| set | paired decks | D5 | candidate | paired difference ± 1 SE |
|---|---:|---:|---:|---:|
| New validation | 113 | 48.7% | 54.0% | +5.3 ± 4.1 points |
| Old regression subset | 99 | 42.9% | 46.0% | +3.0 ± 2.7 points |

SE is clustered by deck family; each regression deck averages its two seeds. One
regression deck was excluded because a fight capped. Both agents completed all new
validation starts. Improvements are promising but do not clear the promotion gate;
**D5 remains selected and its original checkpoint is unchanged**.

Cached MCTS20k reference on the same 99 regression decks: **55.6%**, versus candidate
46.0%; candidate-minus-MCTS **−9.6 ± 3.5 points**. We have not beaten MCTS. This is the
existing benchmark teacher run, not new MCTS collection on the validation pool.

Candidate for inspection:
`runs/schema=combat_v4/date=2026-10-05/id=human-combat-r01/out/iter000/model/`.
Detailed result: real run `out/REPORT.md`, `iter000/comparison.json`, `mcts-context.json`.
Next recommendation: another fresh human-deck batch with the same architecture/search,
then fresh paired confirmation before promotion, rather than an architecture sweep.
The existing unbounded softplus value head still predicts scores above 100 in some
validation states; that pre-existing contract was intentionally not changed in this round.

All original benchmark caveats remain: reconstructed deck exactness is not historical
state exactness; missing potions/counters and balance changes; reconstruction selection.
Validation labels represent baseline continuation, not oracle optimal values. Success
means improved paired gameplay, not lower training loss or more confident predictions.
