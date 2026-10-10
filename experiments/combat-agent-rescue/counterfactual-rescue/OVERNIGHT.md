# Overnight continuation — 2026-10-06/07

Status: protocol frozen by Astra; implementation/launch review pending. User authorizes
up to eight hours of unattended work. Deadline: **2026-10-07 07:15 BST (06:15 UTC)**.
Do not extend the session or launch work expected to overrun this deadline.

## Outcome before continuation

Current candidate: Phase5 strong update3, learned policy/value + PUCT2k with50% guided
rollout leaf values. Development83/100 vs MCTS74/100. This is not fresh confirmation.
Phase6 first evaluates the frozen candidate vs MCTS on600 untouched reserved starts.
That comparison must finish/report independently; overnight training does not modify it.

## Purpose

Continue the successful rollout-assisted learning mechanism on fresh full combats,
with two training replicates. Seek a stronger stable checkpoint, not a card-specific
hack or a broad hyperparameter sweep. Exact deck/HP/relic/no-potion condition unchanged.

## Frozen training design

- Replicas A/B both start from the exact Phase5 strong-u3 model AND AdamW moments.
- Random seeds1/2 for training samplers; distinct fresh gameplay-seed namespaces.
- Rollout mix0.5,2000simulations,greedy collection and evaluation, same frozen pv_worker.
- Each replica:20updates,300fresh full training fights/update:6000fights per replica.
- No exploration flags, architecture changes, value-root mixing, new rewards or teacher
  trajectories. Actual terminal100/0 outcome values; existing concentration-weighted
  search-visit policy targets, equal loss weights.
- Per update1000optimizer steps,batch64 EXACT32fixed-old+32recent-new; fight-uniform then
  state-uniform sampling within halves. lr3e-4,wd.01,grad clip1,optimizer resumed.
- Fixed-old pool: original1000round6–15 fights +432Stage2rescue fights +600Phase5strong
  fights =2032unique fight IDs. Assert counts/contracts/disjointness, never silently dedup.
- Recent-new pool: latest SIX completed collection batches of the same replica,
  at most1800fights. Earlier rows may remain on disk but are not sampled.
- Save each model/optimizer; no validation-loss checkpoint selection.
- Training processes run **sequentially**, not simultaneously, to cap RAM. Gameplay
  uses at most10workers globally. No overlap with Phase6 gameplay.

## Predeclare splits before outcomes

Freeze ALL seeds/manifests before first overnight gameplay:
-12,000distinct training starts, disjoint across replicas/updates.
-200new MONITOR starts: teacher reference once, and the original Phase5 champion once.
-600new FINAL starts, unused until challenger checkpoint selected and frozen.

Exclude every seed actually used or reserved in the previous relevant experiments,
including Phase5train/smoke, Phase6reserved confirmation, development, original run,
source/selection starts and restart-source combats. Reading seed manifests for exclusion
is allowed; no playing final starts during training or selection. No seed substitutions.

## Monitoring and candidate selection

Evaluate each replica on the same200monitor starts after updates5,10,15,20, greedy2k
mix0.5. Cached monitor teacher/champion are NOT training data.

Primary purpose is choosing ONE challenger for independent final evaluation. Selection
rule frozen now: highest monitor wins among the eight scheduled checkpoint candidates;
ties prefer lower update number, then replica A. Do not use time/noisy loss or prior
final outcomes as tie breakers. Include only fully completed scheduled evaluations.

If none strictly exceeds the original champion's monitor win count, report no promising
challenger; do not spend the reserved final budget by default. Preserve both runs/results.
If a challenger is selected, snapshot exact checkpoint/ONNX/external-data hashes BEFORE
final dispatch. Model selection on monitor is expected; monitor intervals are descriptive.

Safety against severe regression: if a replica scores at least20wins below champion on
TWO consecutive scheduled monitor checks, stop collecting that replica. This is a
predeclared compute-saving rule, not a statistical claim. The other replica may continue.

## Final confirmation (conditional on promising challenger)

On the NEW600final starts, evaluate selected challenger, original Phase5champion and
MCTS20k:1800games, same pairs and frozen worker. No cached references across seed sets.
Predivide manifest into six consecutive100-seed batches for descriptive robustness.

Single primary claim: challenger minus MCTS paired win gap,95% interval and exact
McNemar test. Secondary: challenger minus incumbent. No checkpoint reselection on final.
Report all rates, discordance, intended/completed coverage, mean/median inference time,
training cost, single-machine runtime and all errors/caps. Approximate intervals and
small batch samples labelled honestly. One selected model is not general-deck evidence.
Do not claim reliable improvement over incumbent from a positive point estimate alone.

## Budget, deadline, and failure handling

Maximum gameplay after Phase6:
-12,000training
-400initial monitor reference (teacher+champion)
-1,600checkpoint monitoring (8x200)
-1,800conditional final
=15,800games, plus at most4declared wiring smoke games. Phase6's1200 are separate.

Expected runtime is several hours; estimate from existing measured7–8s/game and actual
training throughput before launch. This is an estimate, not a guarantee. Global deadline
07:15BST. Reserve60minutes for final/report: do not begin a new training update after
06:15BST, and stop earlier if measured ETA exceeds remaining time. Final collection starts
only if projected to finish with15minutes remaining; otherwise leave it unplayed/report.
A deadline stop is not permission to choose an unscheduled checkpoint or call partial
coverage a complete confirmation.

Any process/semantic/encoding/trainer error: stop dispatch, retain full ledger, notify SRE.
Training caps: exclude from labels, count them; fewer270/300completed stops that replica
pending review. Monitor/final caps/errors invalidate complete-comparison claims; no loss
relabeling and no replacement games. No automatic gameplay retries.

## Automation and SRE

Use one resumable stage-driven runner, automatic REPORT/checkpoint status at each stage,
atomic state updates, completion-order per-game ledger and frozen config/source hashes.
Write a dispatch-intent entry BEFORE starting each game. Reuse persisted completed results
on resume; never rerun a completed seed to get a preferable result. Missing outcome for
an already-dispatched game is an incomplete attempt, not permission to replay silently.
A crash during training may resume from the preceding intact checkpoint using the SAME
frozen data/RNG/settings, recorded explicitly; do not substitute a partially written model.

The cheap orch-26-10-6 owns implementation/orchestration/SRE. Astra owns research changes.
SRE may fix disk/logging/process/report issues and resume documented stages without new
research approval. It must not change seeds, model, objective, search budget, loss or
selection rules. Escalate unfixable infrastructure or any research-semantics ambiguity.
No repeated high-context status conversations; use logs and machine-readable state.

Before long launches: announce runtime/goal/log/read-only monitor; background with
preserved logs; bg_wait true exit. Process-group cleanup on timeout. No gameplay overlap
between Phase6 and overnight. Memory target<10GB RSS; halt safely on sustained memory
pressure rather than OOM. All builds under build/<name>, originals immutable.

## Deliverable

A concise morning report covering:
1.Phase6confirmation of the original83/100 candidate.
2.Two continued-learning curves, selection and actual budgets.
3.New final results if legitimately reached, otherwise why not.
4.Exact inference/checkpoint paths and reproduction command for best supported agent.
5.What is demonstrated versus suggested, costs, limitations and remaining work.
