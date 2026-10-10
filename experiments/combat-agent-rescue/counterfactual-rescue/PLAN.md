# Counterfactual rescue: experiment plan

Status (2026-10-06, 23:40 BST): original rescue stages completed, followed by forced-action
screening and rollout-assisted expert iteration. The Phase5 candidate scored 83/100 on
development against MCTS 74/100; independent confirmation is pending. See README.md and
RESULTS.md for current evidence, and OVERNIGHT.md for the authorized unattended follow-up.
This document retains the original rescue hypothesis and plan; later frozen phase
protocols supersede its initial stage sketches.

## Goal

Build a combat-learning recipe that reliably beats deployed MCTS20k on the fixed
Demon Form / Runic Pyramid vs A20 Champ task at practical single-machine cost.
Generalization across decks comes later. No finite evaluation guarantees superiority
on every possible batch of 100 combats.

## Hypothesis

Shallow learned search can reinforce its own strategic blind spots. Ordinary self-play
may not explore useful alternatives or learn to execute the continuations they require.
Explicit action exploration plus backward restart training may break that loop.

Two mechanisms must remain distinguishable:
1. **Counterfactual branching:** force alternative legal actions at the same state;
   estimate their outcomes under a specified continuation policy.
2. **Backward rescue curriculum:** learn to complete a successful trajectory from late
   states, then progressively earlier states. This addresses continuation competence,
   not merely exploration of the first action.

Neither mechanism is assumed to work. Restart success alone is not the goal.

## Existing evidence and assets

- Barricade specialist: 595/600 vs MCTS514/600 on untouched seeds; different task.
- Demon Form update15: 62/100 vs MCTS74/100 on fresh development starts.
- Teacher-fitted priors with original value:64/100; paired gain2pp,
  approximate95% interval -7.2 to+11.2pp. Changed behavior, no established win gain.
- Original model at10k simulations:67/100; gain5pp over2k,
  approximate95% interval -2.6 to+12.6pp; about5x cost.
- Teacher rescued44 of100 selected learner training losses in the correction run.
  These are candidates for rescue training, not independent evaluation data.
- Whole-trajectory correction was weakly absorbed; this project must not simply repeat it.
- Teacher-continuation labels exist for278 learner states,4particles/state.
  They describe teacher performance, not current learner calibration.
- Headbutt known-top order is lost by existing search particle sampling and PV encoding.
  Do not call that sampler an exact public posterior. Do not silently train on altered
  known-order states or expose the hidden draw order.

Relevant roots under runs/schema=combat_v4/date=2026-10-06/:
- id=single-deck-demon-form-v1/
- id=demon-form-teacher-correction-v1/
- id=demon-form-correction-fit-v1/
- id=demon-form-decoupled-policy-v1/
- id=demon-form-budget-teacher-v1/
- id=demon-form-continuation-v1/
- id=demon-form-same-root-v1/ (implementation/build status must be inspected)

See experiments/single-deck-expert-iteration/ for detailed prior protocols/results.
Do not repeat their analysis or regenerate already available evidence.

## Stage 0 — minimal machinery and frozen pilot

Inventory existing restart tooling; reuse it rather than rewrite. Inspect source and
actual binary/build status. Preserve originals and the extensive dirty working tree.

Freeze12 of the44 teacher-rescued TRAINING starts by deterministic sampling before new
outcomes. For each, identify three nonterminal teacher-trajectory restart states:
late, middle, early, measured by player-turn progress. Deduplicate short fights and
report unavailable states. Selection must not use new learner outcomes.

Choose states without unresolved known-order information where possible. Report excluded
coverage. If no valid useful curriculum can be built, stop and propose the minimal
history-aware sampler change; do not silently discard the limitation.

Gates: replayed public state/legal actions agree; true-start learner reproduces frozen
worker actions/visits; terminal and selection-task handling works; particle identities
and outcome semantics recorded. No full simulator audit or unrelated refactor.

Deliverable: one short protocol with exact sample manifest, commands, runtime, budget,
logs, tests, and any material blocker. Review before substantive compute.

## Stage 1 — find the learnable boundary

On the12 pilot trajectories, evaluate original update15 learner2k and deployed teacher
from the SAME restart particles at late/middle/early states. Four particles/state is a
coarse screen, not a precise per-state probability estimate.

Nominal budget:12 starts x3 positions x4 particles x2 policies =288 continuations.
Maximum320 continuations including smoke/repeats. Initial reproduction-game cap12 was
explicitly amended to15 after five failed new-app attempts; all attempts count. Actual
reproduction ledger:5 frozen-worker games +5 failed app attempts +5 corrected app games.
Teacher smoke:8 continuations. Main Stage1:288 completed continuations.
Reuse valid existing results when exactly compatible; duplicates are not independent data.
No automatic expansion if states fail or confidence intervals are wide.

Question: does the learner reliably finish late teacher positions but fail earlier?
- If yes: initialize the backward curriculum around that transition.
- If no: test a simpler late boundary or identify a concrete state/action representation
  problem. Do not launch broad sweeps.
- If learner already succeeds throughout: this pilot offers little curriculum signal;
  examine learner-path disagreements instead, without claiming teacher paths solve it.

This stage locates a training starting point. It is not a test of general superiority.

## Stage 2 — one controlled rescue-training pilot

Detailed protocol/budget must be frozen after Stage1, before training. Compare:
- **Control:** original update15, additional ordinary learner replay/training.
- **Rescue:** same initialization and optimizer rules, with a declared allocation to
  restart/branch examples; retain ordinary replay to limit forgetting.

Match optimizer-step and sampled-state budgets. Report extra generation cost separately;
matching gradients does not match compute. Use a fixed training endpoint, save intermediate
checkpoints, and do not select by the consumed monitoring wins.

Training design:
1. Begin at the earliest teacher restart region the learner can execute consistently.
2. Collect complete learner continuations from public-belief samples, retaining wins AND
   losses, caps/errors separately. Move starts earlier using an explicit success threshold
   and minimum number of NEW probe continuations; do not reuse training wins as mastery.
3. Where progress stalls, branch at a limited set of meaningful decisions. Guarantee
   exploration of learner-search action and teacher-proposed alternative at the SAME
   state. Other actions can enter only within a frozen candidate/budget rule.
4. Sample the root uncertainty BEFORE forcing each action. Use completed continuations,
   not the same shallow value estimates, for the action comparison. Teacher proposal
   and action evaluation samples must be separate. Preserve all evaluated outcomes.
5. Train policy on action-comparison evidence, with a specified uncertainty-aware target;
   do not force a winner from statistically indistinguishable estimates. Document the
   exact loss before collection. Value outcomes remain tagged by continuation policy.

No global card bonuses/bans, success-only filtering, or clairvoyant state features.
Do not combine architecture, objective, exploration, and curriculum changes in one sweep.
Policy/value trunk drift must be measured; use fixed-source evaluation when necessary
for attribution, not as an unreported agent change.

## Stage 3 — does it improve complete fights?

Evaluate control and rescue at2k from original starts, never from privileged restarts.
Use the existing100 development starts and cached MCTS74/100 reference. Report paired
wins/gaps/intervals, discordance, incomplete coverage, mean/median runtime and generation
cost. Development results guide decisions; they are not final confirmation.

Only a promising fixed recipe proceeds to a second training RNG and fresh confirmation.
Keep existing reserved600final seeds untouched until checkpoint/recipe frozen. A convincing
claim requires superiority to MCTS, not just improvement over62/100 or pilot memorization.

## Stop/change rules

- Failure of replay/hidden-information gates: halt collection, fix or narrow explicitly.
- Better restart fitting without improved full fights: no automatic data expansion;
  assess curriculum transfer/action-ranking mechanism.
- No rescue-specific improvement over matched control: report a failed pilot. Choose
  one justified mechanism change, not more epochs by default.
- Runtime/memory exceeds approved budget: stop and report, no automatic retries.

## Ownership, compute, and communication

Astra owns experiment decisions. Initial executor counter-factual-rescue was replaced
by single-fight-opus after stalled implementation. User subsequently authorized
orch-26-10-6 as cheap overnight orchestrator/SRE. Current ownership is in README.md.
Stage budgets are approved by Astra, not new user permission for routine commands.
Execution agents own implementation, data gathering and factual reporting; strategic
interpretation and intervention design remain with Astra and user.

- Max10gameplay workers globally; bounded RAM, no DuckDB fanout.
- CMake builds only under build/<name>; originals/checkpoints immutable.
- Results, source snapshots and checkpoints in managed runs/, not source directories.
- Before >1min command: runtime estimate/uncertainty, goal, log and read-only monitor;
  background logging and bg_wait completion check. Bundle commands into meaningful stages.
- One stable handoff referencing these docs, not repeated full-context messages.
- One compact report per stage: result, interpretation, budget/status, artifact paths,
  next decision. Immediate messages only for material blockers/safety issues.
- Do independent light work during compute; no conversational polling or artificial
  keepalive messages. Arrange meaningful interactions within roughly5minutes when natural,
  but do not split experiments, waste tokens or distort compute to chase cache expiry.
  Cache hits depend on provider/harness behavior and are not guaranteed.
