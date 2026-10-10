# Proposed Champ corpus experiment

**Superseded for initial execution by [CORPUS_PILOT.md](CORPUS_PILOT.md)**, the four-round pilot
with original decks replay-only. This file remains a possible later scale-up, not current launch instructions.

Status: discussion draft, NOT preregistered, implemented, or launched. Counts below are proposed
budget choices, not empirically optimized hyperparameters. Freeze manifests and protocol before execution.

## Question and scope

Does expanding rollout-guided expert iteration to many human deck families produce a single
A20 Champ agent that beats MCTS20k on unseen families? Start from the tested ten-deck update006
checkpoint (see TEN_DECK_RESULTS.md), not an untested later checkpoint.
Keep width64, 2k simulations, rollout_mix=.5, original HP/relics, no potions, actual-win value
targets, visit policy targets, existing policy weighting, collection exploration and turn sampling.
Keep the compatible frozen worker/encoder. No architecture/search changes in this experiment.
The claim is about the supported human-start distribution, not all characters/bosses or all HP values.

## 1. Freeze families, not just fight seeds

- Audit historical train/validation/monitor/test use and checkpoint ancestry before sampling.
  Prior notes list 1,028 pool decks and 540 held-out families; these are historical counts,
  not a newly verified inventory. Existing split tooling is a starting point, not proof of no leakage.
- Training: 200 new families plus eligible legacy training families/data.
- Development: 30 separate families, 40 paired seeds each. Used repeatedly for selection.
- Final: 100 separate families, 40 paired seeds each. Never inspect model outcomes until
  one checkpoint is selected and frozen. No training or checkpoint selection on these families.
- Split before collecting new trajectories. Group duplicate card/upgrade configurations and
  related snapshots from the same source run where provenance permits. Audit family fingerprint
  semantics; different UUIDs alone do not establish independent decks.
- Deterministic, composition-stratified sampling with fixed quotas across deck mechanisms;
  publish the quotas and resulting target distribution. Do not filter on MCTS or model win rate.
- Call them human-derived starts unless exact-start human wins are documented. A human winning
  a run, using potions, or fighting at different HP does not prove this reconstructed start winnable.
- Preflight simulator AND encoder support. Publish exclusions and a deterministic replacement
  list before outcomes; do not remove hard/stalling decks after seeing results.

## 2. Inventory and reuse before collection

Create an immutable manifest of eligible legacy trajectories and encoded shards, with family,
full start identity, environment seed, search RNG/configuration, worker/encoder/model hashes,
collection round, outcome and original split role. Deduplicate semantic fights, not just filenames.

- Reuse compatible training trajectories and encoded rows; preserve actual outcome labels.
- Outcomes-only caches can save baseline evaluation, but cannot supply visit-policy trajectories.
- Keep all historical monitor/test trajectories out of training for this protocol.
- Cache MCTS references once per exact start/seed/worker/search configuration. MCTS screening
  games are reusable as train data only on training families with suitable recorded trajectories.
  Selected screening seeds are not a fresh final test.
- Keep producer metadata: old learner outcomes estimate old behavior, not the current policy.
  Reuse as controlled replay; do not assume old and new data are interchangeable.
- Resume checkpoint and optimizer with explicit lr=3e-4; verify optimizer/lr restoration.

## 3. Ten bounded collection rounds

At each round admit 20 new training families in a pre-fixed balanced order:

| collection | proposed quota |
|---|---:|
| MCTS20k bootstrap on new families | 20 × 50 = 1,000 |
| incumbent learner-search on new families | 20 × 100 = 2,000 |
| incumbent learner-search on previously admitted/legacy training families | 3,000 |
| total | 6,000 fights/round |

Fill new-family teacher quotas with compatible existing training games first. Use fresh learner
seeds. Freeze the collecting checkpoint within each round; train once after collection completes.
Choose old families with uniform shuffled coverage, not proportional to their existing fight counts.
Use no adaptive hardness weighting initially: noisy win rates and variable fight costs can waste
budget on outliers. A uniform floor plus capped adaptive allocation can be a later ablation.

Proposed bounded training update: 10,000 sampled fight slots × 64 states/fight × 3 epochs,
lr=3e-4, fixed batch size matching the incumbent recipe. Sample 20% teacher, 40% current-round
learner, 40% historical learner. Within each source: family uniformly, then fight uniformly.
Sampling may repeat fights; this is a replay distribution, not 10,000 unique games. Keep all
eligible data on disk; bounded sampling avoids increasing update cost with the corpus size.

This modest recency preference is a hypothesis, not an established improvement (the earlier
single-deck recency experiment was inconclusive). It prevents legacy ten-deck volume dominating,
while maintaining old-family coverage. Never upweight wins or relabel old losses.
Log unique fights/families, realized sampling weights, optimizer steps and training time.

After round 2, operational gate: support, resume, memory, throughput, replay coverage and dev
regression. Do not infer a generalization trend from one noisy check. At round 4, pause expansion
if dev mean paired gain has failed to exceed round 0 at both rounds 2 and 4, or is worse by >5 pp
at both. This is a resource heuristic, not a significance test. Diagnose rather than automatically
expand through a plateau. Any changed recipe becomes a documented protocol amendment.

## 4. Sparse assessment and final decision

- Development at rounds 0,2,4,6,8,10; same 30 × 40 starts, greedy evaluation.
  MCTS reference once. Round 0 measures incumbent zero-shot ability on these families.
- Select highest dev macro-average paired gain among these checkpoints; ties favor the earlier
  checkpoint. Dev results are selection-consumed and descriptive, not confirmation.
- Test selected checkpoint once against MCTS20k on the locked 100 × 40 final starts.
- Primary endpoint: mean over families of paired win-rate differences. Report 95% family-cluster
  bootstrap CI (resample whole families, retaining within-family paired results). This represents
  sampled-family variation, not training-RNG uncertainty. Also report conditional paired seed
  uncertainty for the fixed benchmark, per-family results, median gap, and fraction of decks ahead.
- Superiority criterion: primary 95% CI lower endpoint >0. Practical target: mean gain >=5 pp.
  Failure to clear either is reported plainly; no extra final sampling after inspecting the result.
- Per-deck 40-seed estimates are noisy: this design targets breadth, not proof of superiority on
  each deck. Precision/power cannot be promised before estimating between-family variability.
- Report results with/without the largest-gain deck as a sensitivity check, not a replacement endpoint.
- Track inference CPU/wall time and training cost separately. 2k simulations is not necessarily
  ten times faster than 20k: NN inference and guided rollouts have different costs.
- Infrastructure errors are not losses: repair/retry identical jobs transparently. Agent-induced
  turn-cap failure is a failure to win under a common fixed rule. Report all statuses and coverage;
  unresolved infrastructure failures prevent a clean primary result, rather than silently dropping pairs.

## 5. Budget (planning arithmetic, not a timing measurement)

| work | games |
|---|---:|
| training collection: 10 rounds | 60,000 maximum new; less with teacher reuse |
| development learner: 6 × 30 × 40 | 7,200 |
| development MCTS reference | 1,200 |
| final: 2 × 100 × 40 | 8,000 |
| total assessment | 16,400 |

At the user's approximate 15 core-seconds/game, training collection is 250 core-hours and
assessment 68.3 core-hours: 318.3 core-hours total, or an idealized 31.8 hours at 10 workers,
BEFORE encoding, optimization, export, I/O and stragglers. No measured timing sample size or
uncertainty was supplied. Prior library notes report large deck-dependent timing differences;
this is not a wall-time promise. Re-estimate separately for MCTS and learner after the first round.

The nominal training:assessment gameplay ratio is 3.66:1, consistent with the previous >=3:1
compute preference only if costs are similar. Measure actual CPU/worker-seconds and optimization
cost, and cap intermediate monitoring first if needed. Cached games cost zero new compute;
teacher reuse can lower the incremental ratio. Do not add collection just to satisfy a bookkeeping
ratio: reduce monitoring or agree an exception before running. Smoke/retention diagnostics need
an explicit small allowance; no unbudgeted full incumbent final arm.

## 6. Minimal implementation work before launch

Extend the combat expert-iteration app, reusing corpus split/grouped-sampling helpers rather than
reviving the old corpus recipe. Current app assumes a fixed deck list, starts with fresh bootstrap,
and its runbook says it has no resume: it does NOT yet implement this plan.

Required: init-checkpoint mode, immutable family/split/data manifests, admission rounds, bounded
family/source-balanced replay, persistent cache keys, completion-order per-fight journaling,
idempotent resume and atomic checkpoints, immutable evaluation namespaces, compute accounting,
and a single status command. Keep final play behind an explicit selected-checkpoint lock.
Test split overlap/duplicate rejection, source weighting, stale-cache rejection, resumed jobs versus
uninterrupted jobs, and monitor/final exclusion. Small compatibility smoke before any long launch.

Open approval points: intended human-start population, compute cap, replay proportions and round
quotas. Freeze concrete family manifests, seed namespaces, hashes and stopping/selection rules
before calling this preregistered.
