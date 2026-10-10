# Recency-weighted hybrid continuation — setup

Status: **RUNNING**, approved by user with 10 cores; launched 2026-10-07 20:04 BST. Controller is `apps/run_rl/expert_champ_run.py`, executed from the frozen run copy. Both gameplay smoke fights and encoding passed; incumbent monitoring started. Seed/code freeze: 1,802 distinct starts, 58,744 relevant prior seeds excluded, including old overnight reservations. Twenty-two focused tests passed. Ten native workers observed across cores 0–9. Main log: `runs/schema=combat_v4/date=2026-10-07/id=expert-champ-recency-v1/logs/main.log`. Runtime cap 110 minutes from launch, with a 20-minute final reserve. State/ledger are authoritative for current progress. One isolated two-step trainer smoke passed and is kept outside the candidate path: 64 anchor + 64 online draws; AdamW step 35180 -> 35182; ONNX/PyTorch maximum absolute difference 0.0000229; observed training/export section 7.1 seconds (one smoke, not a pilot runtime estimate). Fifteen focused tests passed. The smoke model is never a promotion candidate.

## Decision

Single training branch, initialized from the confirmed Phase5 strong/update3 weights AND AdamW moments. No duplicate control-training branch. Compare candidates against the frozen incumbent on matched evaluation starts; any gain is a gain from the overall continuation recipe, not causal evidence that recency weighting beats uniform replay.

Keep rollout-assisted neural MCTS: 2,000 simulations, rollout mix .5, greedy collection and evaluation. No architecture, objective, policy-confidence weighting, HP, pile, or card-rule changes.

## Approved pilot (frozen before launch)

- Five updates, each 200 new full fights and 1,000 optimizer steps.
- Same optimizer: AdamW lr 3e-4, weight decay .01, gradient clip 1; resume moments.
- Each minibatch exactly 32 anchor + 32 online states.
- Anchor = original Phase5 old pool (1,000 original learner fights + 432 rescue fights), uniform by fight then state.
- Online = inherited three Phase5 hybrid batches plus new completed batches, retaining wins AND losses. Each fight receives weight `2 ** (-age_in_updates / 1.0)`; state uniform within selected fight.
- One-update half-life is an explicit simple design choice, not an empirically optimized constant. Equal-sized batches make the newest batch at least half the online pool's sampling mass asymptotically (approaching 50% from above), rather than shrinking as 1/number_of_batches. No hard window.
- Fixed training endpoint per update; save/export every checkpoint. Partial checkpoints never become collection models.
- Proposed monitor: 200 fresh starts, incumbent once, candidate every update. Select highest complete win count, ties earliest update; must strictly beat incumbent to spend final budget.
- Proposed final: selected candidate and incumbent on 600 fresh paired starts, no reselection. Existing MCTS confirmation already establishes the incumbent reference; no new MCTS arm needed for this initial improvement question.
- Freeze exact budget, exclusion manifest, sources, selection rules and runtime/deadline before launching. Monitor results are selection-consumed, not independent confirmation.

## Why no augmentation yet

Changing HP can produce legitimate new training starts if the simulator supports them and fresh search/outcome labels are generated. Changing draw/discard membership is not label-preserving: it changes card availability/history and can create inconsistent or unreachable states. Shuffling an actually unknown draw order is a different operation and existing public-belief particle machinery already does this during search.

Do not perturb recorded inputs while retaining old visits or win/loss labels. State augmentation is a separate future collection experiment, not bundled with recency weighting.

## Code ownership

- `agents/combat/pv/recency.py`: fight-balanced age weights and sampler.
- `agents/combat/pv/train_recency.py`: fixed-step trainer, checkpoint/hash validation, optimizer resume, exposure reports and ONNX parity.
- `apps/run_rl/expert_champ.py`: setup/inspection app (currently prepare/status only; deliberately no run command).
- `apps/run_rl/expert_champ_status.py`: read-only dashboard.
- Tests beside owners: `agents/combat/pv/test_recency.py`, `apps/run_rl/test_expert_champ.py`.

Existing phase5/overnight collection and ledger machinery can inform the controller, but must be reviewed rather than launched unchanged (old paths, two-replica settings and fixed historical deadlines do not fit this pilot).

## Dashboard

```sh
watch -n 5 '.venv/bin/python -m apps.run_rl.expert_champ_status --run runs/schema=combat_v4/date=2026-10-07/id=expert-champ-recency-v1'
```

The original `single_deck_status` expects a different pipeline (fresh bootstrap / teacher stages / epoch logs), so do not point it at this run. The new dashboard preserves the same interface style while displaying recency, fixed-step progress, incumbent comparisons, caps/errors and unresolved intents.

## Launch gates (completed / runtime enforced)

1. Audit all previously used and reserved seeds (including unused overnight reservations), generate and freeze train/monitor/final splits.
2. Implement and test single-controller lock, write-ahead dispatch ledger, completion-order persistence, fail-closed resume, subprocess cleanup, timeout/deadline and worker limits. Controller: `apps/run_rl/expert_champ_run.py`; tests beside it. No automatic gameplay retries or partial-checkpoint promotion.
3. Freeze exact runtime sources/imports, verify raw collection->encode->trainer manifest chain, and run bounded gameplay smoke.
4. Announce runtime estimate, log path, global worker count, and proposed deadline before actual pilot launch.

Existing unrelated git changes remain untouched. Focused tests passed; repository-wide diff check reports pre-existing trailing whitespace in `experiments/champ-oracle-exit/RUNBOOK.md`, not changed in this setup.
