# Slime Boss: first value-search improvement experiment

## Approved implementation / pilot revision

User approved the first implementation slice after design review: light separation into `agents/teacher_leaves.*` (strategies) and `teacher_search.*` (recording), shared hybrid/immediate NN evaluation loop, exploration toggle with unchanged bootstrap defaults, explicit validation-episode subsets, safe isolated build and per-run binary snapshots, backward-compatible worker CLI. No simulation-budget override or external-cohort support in this first slice. Implementation: impl-agent; builds/execution: terminal-agent.

**This supersedes Stage 1 below:** the pilot will compare fresh R/N/H on all 14 validation fights with exploration OFF for every arm. First run episode 707 for each arm as a smoke test, then obtain coordinator go-ahead for the remaining fights. No exploratory stored teacher is used as the clean baseline. This adds 14 baseline evaluations to the original resource envelope (about 262 total if every later stage proceeds); actual cohort counts/timing will be frozen before later approval. Stages 2–3 remain future proposals, not current implementation authorization.

The earlier two-fight immediate-NN smoke passed start-state checks and completed in 7.3/29.1 seconds including replay; both won but retained less HP than stored exploratory teachers. These are diagnostic observations only. Artifacts: `scratch/valexp_smoke_20260923T204034/`.

## Decision and scope

Coordinator recommendation: freeze `value_net_v1/2026-09-23/slime-value-4`; measure immediate neural search, then test ONE short-rollout hybrid. Do not retrain, sweep architectures, or start another data-generation job in this experiment. Implementation can be delegated; expensive execution requires the owner's resource approval.

Hypothesis: a short guided rollout before neural evaluation can reduce harmful leaf-value errors enough to improve fight outcomes. This is a hypothesis, not a demonstrated diagnosis. Success over the rollout teacher is not guaranteed.

## Evidence inspected

SQL agent's snapshot: `scratch/slime_boss_explore_report.md`; reproducible queries: `scratch/slime_boss_explore.sql`. Snapshot labeled 2026-09-23T19:37Z; live counts grew during queries, so these are not an immutable cohort manifest.

| combat_v3 run (2026-09-23) | Slime fights | Teacher wins | Relationship to checkpoint |
|---|---:|---:|---|
| act1-a20 | 69 | 46 | 55 training, 14 validation |
| act1-a20-1 | 20 | 14 | unseen seeds |
| act1-a20-2 | 78+ | 53/78 at snapshot | unseen seeds; generation running |

- No overlapping run seeds across those three runs; one Slime fight per seed. Source runs -0/-1 were interrupted, so use only complete, validated fights and replayable preceding fights.
- Existing validation set: 12/14 teacher wins; it is a small, unusually successful sample, not representative evidence of a saturated task.
- 167 snapshot fights had 167 distinct card-ID deck multisets; starting HP ranged 30–85. Rows within a fight are not independent samples.
- Checkpoint trained on 7,527 rows total across train/validation: 1,744 decision + 5,783 child rows. Training MAE .03298, validation MAE .09252; MSE .002355 vs .014088. This suggests a generalization gap, not a diagnosis of action-ranking quality.
- No current combat_v3 value-play/comparison results were found. Older gen0/combat_v2 experiments are different experiments and must not be reported as this checkpoint's performance.
- Teacher root values are lower than realized returns on average after the exploratory move, especially early in fights. This is compatible with rollout-continuation pessimism but NOT proof: root values average explored actions; realized returns are noisy; grouping by eventual victory induces conditioning bias. Search and stored outcomes also have slightly different scoring.

## Current implementation constraints

Relevant files:
- `agents/teacher_search.{hpp,cpp}`: shared search/recording, 15,000 simulations, 8 particles, neural batch 64, max_actions 512, early stop; one exploratory move in decision indices [0,24) if reached.
- `apps/value_play/{play.py,worker.cpp,run.sh,job.sh}`: currently only checkpoint validation fights from its sole training-source run; shared recorder injects exploration here too.
- `apps/compare_fights/compare.py`: terminal_value primary, paired bootstrap CI, Wilcoxon, exact McNemar for wins. Currently candidate-driven pairing can silently omit unsuccessful candidate jobs unless an external expected cohort check is added.
- `../sts_lightspeed/src/sim/search/PublicBeliefCombatSearch.cpp`: bounded rollout support exists; neural wrapper currently calls `requestBatch(batch, budget, 0, 0)`.

Search's native terminal score includes a tiny turn penalty and nonzero loss shaping, unlike stored terminal_value. Keep this behavior FIXED across this ablation, document it, and test terminal handling. Do not silently change objectives while swapping evaluators. Exact terminal leaves must remain exact; do not run the NN on them. Retain current network clamp for this experiment.

## Stage 0 — freeze inputs and validate plumbing (cheap)

1. Preserve live bootstrap processes, config, and binaries. Working tree is dirty/shared; make narrow edits, no resets or broad refactors. Use a dedicated experimental build directory so compilation cannot overwrite live-job executables.
2. Hash checkpoint and exported weights; capture source revisions AND dirty diff/source snapshot. Record explicit source-run/episode pairs, run seeds, shard hashes, and search settings. Preserve completed shard contents via safe copy/reflink or equivalent; do not rely on mutable directories or compaction-sensitive paths.
3. Check all required preceding fight actions are available and target fights complete. Check A20 and Slime Boss identity. Replayed initial encodings must match source encodings.
4. Freeze the existing 14 validation episodes as PILOT only. No changes to training split/checkpoint.
5. From all currently complete unseen Slime fights in act1-a20-1/-2, assign 24 DEVELOPMENT fights by deterministic seeded shuffle (record algorithm/seed and resulting manifest); reserve the remainder for CONFIRMATION. Do not select based on outcomes, decks, or regression severity. All splitting is by run_seed. At the initial snapshot this leaves about 74 confirmation fights. Later arrivals may supplement confirmation under a predeclared rule, never by results.
6. Implement tests below before substantive runs.

## Minimal delegated implementation

Extend the existing replay app rather than creating a second search implementation:
- Optional explicit evaluation data sources and episode manifest, independent of checkpoint's training source; defaults preserve existing validation replay behavior. Record all inputs in run provenance.
- Per-run search configuration: leaf mode `guided_rollout`, `value_net`, `hybrid`; simulations; batch; rollout turn/action bounds; exploration on/off. Names illustrative, not a required API. Defaults preserve bootstrap behavior.
- Exploration disabled only in evaluation configuration; bootstrap retains its existing exploration. Record actual settings rather than hardcoded teacher defaults.
- Hybrid: existing requestBatch mechanism with bounds `(1,16)`. This means stop after at most one turn increment OR 16 rollout actions, NOT one complete future player turn. Immediate NN uses `(0,0)`.
- Support fresh rollout baseline on the exact same evaluation manifest and without exploratory moves. Do NOT compare exploration-off candidate directly against stored exploration-on teacher as the clean milestone.
- Per-fight sidecar telemetry: target-combat elapsed time separate from replay/startup, search time if practical, decisions, turns, simulations, NN leaves, exact-terminal leaves, unresolved leaves, settings, completion/error status. Store full attempt manifest even when a worker fails.
- Configurable timeout/decision safety cap. Mark unfinished/error explicitly, never as ordinary combat defeat and never silently drop from comparison. Suggested initial cap: 500 decisions or 10 minutes per fight; revise after smoke timings BEFORE development starts, then freeze equally across arms. A cap hit blocks an unqualified performance claim; show completion rates and sensitivity bounds or resolve with a predeclared larger budget.
- Comparison must verify every expected episode is present exactly once on both sides, with matching start-state identity/provenance. Existing statistics can be reused for complete cohorts; refuse an ordinary complete-cohort report when attempts are missing.

Required tests:
- Defaults preserve current bootstrap settings/behavior; exploration=false emits no was_random rows.
- Immediate `(0,0)` remains original neural path.
- Bounded rollout stops with correct turn/action semantics; terminal rollout completion skips NN and uses native terminal score.
- Batched response IDs map correctly and values finite; existing Python/native parity tests pass if applicable.
- External-source replay matches stored starting state; preceding actions exhausted exactly.
- Cross-split run-seed overlap rejected; expected-cohort completeness catches failed/missing workers.

## Stage 1 — 14-fight compatibility pilot

Run immediate NN and hybrid on the existing 14 validation fights WITH original exploration, compare each against stored teacher outcomes. This is the cheapest baseline because teacher fights already exist. Keep 15,000 simulations, 8 particles, batch 64, current early stopping and forced-move budget; vary only leaf mode/bounds.

- First smoke both modes on two manifest-selected fights; if healthy, complete all 14. Start with one evaluation worker, not a full CPU pool alongside the ten-worker generation job.
- Report outcomes and timing, but do not make significance/promotion claims from n=14.
- If mismatch, invalid value, pathological runtime, or missing fight: fix plumbing or stop before spending on the main cohort.
- Retain this as a compatibility check; exploration-on results do not establish clean policy strength.

## Stage 2 — clean development ablation (24 fights)

Replay three arms on the SAME 24 frozen development fights, all exploration OFF:

| Arm | Evaluator | simulations | particles | neural batch |
|---|---|---:|---:|---:|
| R | full guided rollout | 15,000 | 8 | n/a |
| N | immediate NN `(0,0)` | 15,000 | 8 | 64 |
| H | short rollout -> NN `(1,16)` | 15,000 | 8 | 64 |

No checkpoint changes, second hybrid depth, label sweep, or reward change. Compare H-R, N-R, and H-N descriptively; this is selection/development, not confirmation. Equal simulation budgets do not imply equal work, search depth, or identical batching behavior; record time.

Selection gate: choose at most one neural candidate for confirmation, preferring higher mean paired terminal_value subject to runtime/completion and win-rate regressions being acceptable to the owner. If neither improves on R, STOP for diagnosis rather than running a large losing comparison. A small development mean advantage is motivation to confirm, not evidence of success.

Diagnostic fallback if both fail: inspect five worst regressions plus five deterministically sampled non-regressions; generate compact timelines (HP, turn, boss/split HP, chosen action, search statistics) and characterize neural leaf coverage. Decide between targeted relabeling and another search experiment only after that review. Do not automatically retrain on the losing player's predictions.

## Stage 3 — untouched confirmation and claim

Freeze selected candidate/config, cohort, resource budget, and primary analysis BEFORE running confirmation. Rerun clean rollout teacher and the single selected candidate on reserved fights; no exploration. The existing validation and 24 development fights are excluded. Do not evaluate both neural variants and choose the winner on confirmation.

- Primary: mean paired terminal_value difference, with paired bootstrap 95% CI. Keep existing Wilcoxon as supporting evidence (it does not directly test the mean); report wins/discordance and exact McNemar too. No claim based on a secondary p-value alone.
- Require lower primary CI > 0 for evidence of improved terminal-score play. Separately state whether wins improved, fell, or are inconclusive. Terminal score trades win probability against surviving resources; it does not guarantee win-rate superiority. Agree a win-rate noninferiority margin with owner before confirmation if making a safety/noninferiority claim.
- Report n, all attempts/caps, all runtime, exact settings, and gains/losses. One fight per seed currently permits per-fight resampling; cluster by run_seed if that changes.
- Roughly 74 fights is an available fixed cohort, NOT a power guarantee. Use development paired-difference variance and an owner-chosen worthwhile effect to size confirmation before viewing its outcomes; rough 80%-power planning approximation n ~= 7.85*sigma_diff^2/delta^2. Treat it as approximate for this mixed zero-inflated score. If available n is insufficient, accept an exploratory/inconclusive result or wait for already-running generation; do not repeatedly peek until significant.
- This first experiment establishes equal-simulation results. Do not claim equal-compute superiority. For practical improvement, run a separately frozen, dev-calibrated matched-time comparison if needed; measure without uncontrolled generator contention. Never tune budget using confirmation outcomes.

## Resource envelope / owner approval

At snapshot size: pilot 28 candidate fights; development 72 fresh fights; confirmation about 148 fresh fights = approximately 248 evaluations maximum before any matched-time follow-up. Baselines in clean stages must be rerun because stored teachers explore. Smoke fights count toward pilot, not extra jobs.

No reliable runtime estimate yet: time two smoke fights first and extrapolate conservatively, recording replay overhead and variation. Default one evaluation worker while bootstrap runs. Owner chooses whether to schedule overnight, pause competing work themselves, or defer evaluation. Do not stop the generation job on their behalf.

## Deliverables / done criteria

1. Minimal reviewed implementation + tests, with unchanged bootstrap defaults.
2. Immutable manifests/configs/checkpoint hashes and a concise command sheet with actual executable commands after options are implemented (do not pretend proposed flags exist).
3. Pilot and development comparisons, runtime estimate, and explicit recommendation: stop/diagnose or confirm ONE candidate.
4. Confirmation report only if approved and the gate passes. Report failure/inconclusive honestly.
5. No training generation two until a stronger player has been demonstrated or a specific data failure is established.
