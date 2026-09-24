# App specification: apps/dagger/ (approved generator only)

## Purpose and boundary

Collect teacher-corrected learner experience for Slime Boss. A frozen neural-search learner chooses and executes actions; independent rollout-teacher search labels the same pre-action states. Output is combat_v3, not a new schema. This app does NOT train, promote models, or launch another generation.

Design ownership: coordinator. Implementation agent executes this specification and asks before scope changes.

## Algorithm

For each selected eligible fight:
1. Reconstruct its initial state from an explicit source combat run: existing act1 setup and exact replay of preceding stored actions.
2. Verify initial state matches source encoding/identity using the existing replay checks.
3. At each reached decision, independently search the unchanged state with frozen learner NN-MCTS and guided-rollout teacher. Do not share mutable search trees. Existing deterministic public-observation particle seeding is retained; independent here means separate searches, not statistically independent randomness.
4. Record teacher root/edge statistics and actual learner-selected action, then execute ONLY that learner action.
5. Repeat until terminal or a configured safety limit.

Defaults: immediate value-net learner, guided-rollout teacher, existing budgets (15k simulations, eight particles, NN batch64, early-stop and forced-action rules). Learner exploration OFF. No search-leaf harvesting, teacher interventions, random-action injection, child rows, policy mixing, uncertainty sampling, or teacher-budget sweep in v1.

Teacher root_value retains its existing visit-weighted explored-edge average semantics, NOT max-Q or chosen-action Q. Do not change search objectives in this app.

## Eligibility and source provenance

- Input is a frozen value_net_v1 learner run and explicit bootstrap combat source runs (default resolve learner lineage where unambiguous).
- Only checkpoint train_episode_ids AND train_run_seeds are eligible. Validation IDs/seeds forbidden. Explicit additional excluded seeds supported for previously reserved/evaluated sets.
- For the initial slime-value-5 config also exclude all 14 old slime-value-4 validation episodes/seeds (10 were already included in v5 training; excluding now does NOT retroactively make them held-out for v5).
- Optional explicit episode list, or deterministic seeded selection of a bounded number of eligible episodes; never choose by outcome. Reject duplicates/outside IDs, source ambiguities and missing preceding actions. Record exact selected episodes/seeds before workers launch.
- Do not load every combat_v3 run via wildcard: learner replay runs share this schema. Use explicit source lineage and reject known evaluation/DAgger source runs for reconstruction.
- Freeze weights, worker binary, config, selected source data or replay requests/start-state records; record hashes. Live bootstrap directories may grow: select only complete stable records; hash/copy checks must detect files changing during capture. No modifications to live source data.

## combat_v3 row meanings

All output rows have row_kind=decision and parent_action=-1.
- Encoded state and decision_index: learner's actual pre-action state.
- actions/root_value/simulations_used: teacher search statistics.
- chosen_action: action actually executed by learner (need not equal teacher recommendation).
- was_random=false.
- Outcome fields: actual terminal learner fight outcome, using existing terminal_value formula.

No NN self-predictions are used as teacher labels. Root labels are useful estimates, not ground truth. No child rows in v1. If useful teacher recommendation or learner compute counters are not already represented, put them in an auxiliary diagnostic artifact rather than modifying combat_v3 columns for this first version.

## Run/database representation

Keep existing hive layout: schema=combat_v3/date=.../id=dagger-gen1-slime-v5-001. Do not add another hive partition axis. IDs are readable labels, never parsed to infer training semantics.

Record machine-readable provenance in run summary and an initial out/collection.json written before collection:
- collection_method="dagger"
- training_target="teacher_root_only"
- learner run ID, frozen checkpoint/weights hash, learner search settings
- teacher kind/settings and root-value definition
- explicit source run IDs; source hashes/selected episode+seed list
- exploration false, limits, config, worker hash, generation label if provided
- attempted/completed/capped/failed counts and per-fight statuses/times/learner+teacher simulation counts

Every output Parquet file retains schema=combat_v3 and adds collection_method=dagger and training_target=teacher_root_only metadata. Constant provenance belongs at run/file level, not redundant new per-row columns.

Run inputs include the learner run and all used source combat runs. DAgger runs of the same source episode remain distinct run IDs; never overwrite earlier generations or deduplicate them by episode alone across runs.

## Training safety (NOT training implementation)

The current blended trainer must not silently consume this data. Add a minimal load_rows metadata guard: DAgger/teacher_root_only parts raise a clear unsupported-corrective-source message under the current ordinary input path. Add a test. This is a compatibility guard, not a new training feature.

Future work, separately authorised: extend apps/value_train with explicit corrective inputs, teacher-root-only targets for corrections, original blend for bootstrap, controlled source mixing, and pinned seed grouping across sources. Do not create a second trainer or implement these features now.

## Operational design

App structure follows existing apps: run.sh, job.sh, generate.py, worker.cpp, an example Slime TOML, documentation.

Reuse existing search and replay helpers. Extract a small shared helper only where needed to avoid duplicating replay/search-stat mapping; no broad refactor. Existing apps and bootstrap defaults remain unchanged.

Safe isolated build directory (build-dagger), per-run executable/weight snapshots. Default workers=1 to coexist with generation. No builds in live build/ or shared build-dev.

Configure max_decisions (default500) and per-fight subprocess timeout (default600 seconds). A limit/error is NOT a terminal defeat:
- completed fights write atomic combat_v3 part files, one per episode;
- incomplete fights have status/error details in auxiliary manifest, and NO training Parquet output in v1;
- preserve partial diagnostics separately if cheap, not in out/*.parquet;
- run summary states all selected/attempted/completed/missing episodes; non-success collection status on failure/limit, while explicitly preserving valid completed parts.

These conservative incomplete-output rules avoid nullable/fabricated outcomes and keep combat_v3 semantics unchanged.

## Example config contract

[run]
id = "dagger-gen1-slime-v5-001"
input = "value_net_v1/2026-09-23/slime-value-5"
workers = 1

[collection]
count = 100
seed = 0
# episodes = [...]  # alternative to count; reject both specified
exclude_run_seeds = [...] # example config enumerates original v4 validation seeds
max_decisions = 500
timeout_seconds = 600

Resolve source runs from learner lineage; allow explicit source override only with equivalent validation. Keep a small bounded config interface; no arbitrary search hyperparameter surface needed in v1. Existing search constants/settings must be recorded accurately.

## Acceptance checks

1. Output matches combat_v3 including metadata and row meanings; all rows decisions, no random moves.
2. Learner's selected action is executed even when teacher prefers another; collecting labels cannot mutate learner state/RNG/action choice. Separate searches verified.
3. Teacher root/edge values match a standalone teacher search from the same state on a small fixture.
4. Eligibility rejects checkpoint validation, additional exclusions, duplicates, invalid source identity and missing preceding replay.
5. Terminal learner outcomes correct; capped/error fights emit no ordinary training rows and explicit failure statuses; atomic parts.
6. Current training loader rejects corrective Parquet with clear message.
7. Existing bootstrap/replay and teacher_leaves tests continue to pass.

Build/tests and a bounded fixture smoke only once ready. Do NOT launch the configured 100-fight job or training without coordinator approval.
