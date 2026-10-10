# Champ corpus pilot — implementation protocol

Status: agreed pilot direction, recorded after discussion; no gameplay launched. Exact family/seed
manifests and artifact hashes must still be frozen. This supersedes GENERALIZATION_PLAN.md for
initial implementation and the four-round pilot; that document remains a possible later scale-up.

## Goal

Test whether broader training improves unseen-family play within approximately four hours of
interactive time plus eight overnight hours. This is development evidence, not final confirmation.
Keep the final 100-family test untouched. Do not rent additional cores before measuring the pilot.

## Frozen starting point and eligible old data

- Initial model and AdamW state:
  `runs/schema=combat_v4/date=2026-10-09/id=champ-ten-rollout-v1/out/update006/model/model.pt`.
- Same run's bootstrap training trajectories (3,000 games) and learner updates 001–006
  (12,000 games); verify actual completed IDs and hashes when building the manifest.
- Reuse encoded shards where compatible. Keep original targets and producer metadata.
- Do not import update007, monitor/test trajectories, or other historical experiment data for
  this first pilot. Inventory all historical family exposure for split exclusion nonetheless.
- Ten original families are **replay-only for the entire pilot**: no fresh teacher/learner games
  and no unbudgeted retention gameplay. Their existing data remain available throughout.
- This saves compute; it does not prove those decks are mastered. The trade-off is stale
  on-policy coverage and no direct fresh retention measurement. Reconsider after the pilot,
  rather than silently reactivating them in response to noisy collection outcomes.

## Families and starts

- 40 new training families, admitted as four fixed waves of ten.
- 20 development families × 20 paired fight seeds.
- 100 final-test families reserved, with no final gameplay during this pilot.
- Supported human-derived A20 Champ starts, original HP/relics, no potions. No selection on
  MCTS/model win rate, cost, observed success, or failure to win. Do not claim exact-start
  human winnability without supporting records.
- Proposed deterministic composition quotas using existing five buckets (block, demon_form,
  exhaust, strength, mixed): 2/bucket/wave, 4/bucket development, 20/bucket final. Freeze these
  only after a support-only inventory confirms availability; report shortages before outcomes.
  These quotas define an intentionally balanced benchmark, not the natural human frequency mix.
- Deduplicate card/upgrade families and related source-run snapshots where provenance permits.
  Audit fingerprint semantics. Exclude actual historical training exposure, not merely UUIDs.
- Historical manifests are not authoritative: `clash` (9308c94d) was listed as protected in the
  old corpus split, then trained in the ten-deck run. Reconcile actual use before freezing splits.
- Preflight simulator and encoder support; publish deterministic replacements/exclusions before
  outcome inspection. Once admitted, do not drop decks because they stall or lose.

## Collection schedule

Freeze the collecting model within each round; train after collection is complete.
Learner collection keeps incumbent settings: 2,000 simulations, rollout_mix=.5, root exploration
and early-turn sampling enabled. Teacher remains standard MCTS20k, not a win-only variant.

| round | new teacher | new learner | revisit learner | total |
|---|---:|---:|---:|---:|
| 1 | 10 × 20 = 200 | 10 × 80 = 800 | 0 | 1,000 |
| 2 | 200 | 800 | 500 across wave 1 | 1,500 |
| 3 | 200 | 800 | 500 across waves 1–2 | 1,500 |
| 4 | 200 | 800 | 500 across waves 1–3 | 1,500 |
| total | 800 | 3,200 | 1,500 | 5,500 |

Original ten families never enter revisit allocation. Round one's unused 500 is **saved**, not
redirected to new decks. Later revisit allocations are uniform across earlier pilot families:
50/deck in round 2, 25/deck in round 3, 16 or 17/deck in round 4 (seeded remainder allocation).
No hardness, win-rate, or speed weighting. All new starts use frozen train-only seed namespaces.
Teacher and learner seed streams are separate; no monitor/test trajectory enters training.

## Exact replay distribution

Dataset = eligible original training games plus all completed, eligible pilot training games up to
this round. Immutable source shards; do not copy old trajectories into every update directory.
Each epoch has 3,000 fight slots, equally allocated across admitted training families:

| round | original + new families | slots per family per epoch |
|---|---:|---:|
| 1 | 20 | 150 |
| 2 | 30 | 100 |
| 3 | 40 | 75 |
| 4 | 50 | 60 |

**Family first**, then source within that family:
- 20% teacher;
- 40% current-round learner;
- 40% historical learner (strictly before this round).

If current learner is absent, transfer its 40% to historical learner. If historical learner is absent,
transfer its 40% to current learner. Fail closed if teacher or both learner pools are absent: do not
silently drop a family or fabricate data. Source allocations divide exactly for the pilot sizes.

Consequently:
- Original ten: 20% teacher / 80% historical learner, every round.
- Newly admitted wave: 20% teacher / 80% current learner.
- Revisited pilot families: 20% teacher / 40% current / 40% historical.

Within a source sample fights uniformly **with replacement**, then 64 recorded decision states
uniformly with replacement per fight slot. The same fight can occupy multiple slots; it remains
one uniquely identified trajectory. Never duplicate IDs to evade deduplication. Never balance
wins/losses, select winning states, or reward longer fights with more weight.
Resample each epoch from deterministic RNG streams and mix states across families and shards
before minibatching. Log both slot draws and realized state counts by family/source/fight.

Training targets and architecture remain unchanged:
- value = actual terminal win ×100 (not current model value or teacher root score);
- policy = normalized recorded root visits, existing policy-confidence weighting;
- width64, same input contract and sigmoid head;
- three epochs ×192,000 states =576,000 state draws;
- batch64 =9,000 optimizer steps per round;
- AdamW resume, explicitly lr=3e-4, weight_decay=.01, grad_clip=1;
- fixed third-epoch export; no legacy-validation-loss epoch selection. This is an explicit small
  change from the current general trainer. Validation is diagnostic only.

The replay proportions and update size are hypotheses, not tuned or established improvements.
Family balancing gives original decks 50%, 33.3%, 25%, 20% of replay as the pilot grows, despite
having much more stored data. Source balancing applies within families, not globally first.

## Development measurements and decisions

Same 400 starts for MCTS20k and every model; greedy learner evaluation (no collection exploration).
- Before training: starting model (400 games) and MCTS (400, cached once).
- After round 1: candidate (400 games).
- After round 4: candidate (400 games).
- No intermediate round-2/3 gameplay evaluations, no final test or extra incumbent arm.

Tonight's target is baseline plus round-1 comparison. Overnight target is rounds 2–4 and their
endpoint. Compare models by paired outcomes and equal-weight mean family win-rate difference.
Report family-cluster bootstrap intervals, per-family results, completion/status counts, and cost;
20 families ×20 seeds is a noisy development screen, not a definitive superiority test.
One round's failure to improve is not a reason to discard the recipe. Pause immediately for
split/support/resume/data-integrity failures or an untenable measured cost. At round 4, if unseen-
family performance does not improve over the initial checkpoint, diagnose before expanding.
Do not tune on the reserved final set. Any amendment must be recorded before further collection.

Infrastructure failures are not losses and cannot silently remove evaluation pairs. Preserve and
resolve them on identical jobs before claiming complete results. Agent-induced cap outcomes
must be separately reported as failure to win under the frozen evaluation contract; inspect actual
worker status semantics before implementation. Keep the worker unchanged for comparability.

## Time and compute

5,500 training +1,600 evaluation =7,100 new games before any verified cache savings.
At the user's approximate 15 core-seconds/game: 29.6 core-hours, idealized 4.9 hours on six physical
cores, **excluding optimization, encoding and overhead**. No uncertainty estimate or sample size
was supplied for that per-game estimate. Hardware inspected: Ryzen 5 3600X, six physical cores,
twelve logical CPUs. Ten concurrent workers are not ten independent physical cores.

Historical ten-deck logs show six 2,000-game learner batches taking about 23–24 minutes each
(n=6 batches, observed range rounded). This is wall time on those decks, not CPU time or a promise
for the new corpus. Measure teacher/learner throughput separately and training/encoding time in
round 1. Four-hour first result is a target, not a guarantee; implementation and split audit also
consume that window. No long command without an announced runtime, goal and persistent log path.

Full-pilot gameplay training:evaluation ratio is 3.44:1 by game count, not necessarily compute.
The first readout is deliberately evaluation-heavy (1,000 train /1,200 eval); this is the up-front
cost of reusable references. Account across the full pilot and disclose if stopped early. If actual
cost threatens the previous >=3:1 compute preference, agree a revision rather than silently exceed
it or generate unnecessary games just to repair the ratio.

## Implementation boundaries (after inspecting existing code)

Use `apps/combat_expert_iteration/` as the workflow owner; a separate corpus entry point/module
there can keep the established fixed-deck CLI stable. Avoid another competing top-level app.
Keep sampling/optimization under `agents/combat/pv/`, not embedded in orchestration scripts.
Experiment-specific manifests/configurations and analysis belong here or in the managed run;
models, generated datasets, journals and reports belong under `runs/`.

Existing pieces to reuse, with limitations:
- `apps/combat_expert_iteration/expert_iteration.py`: checked starts, frozen worker, command
  construction, encoding, fixed-deck workflow. Currently assumes fresh bootstrap and no resume.
- `evaluate.py`: matched starts, cached result arm, per-deck comparison. Its reuse check only
  verifies IDs/seeds, not the full state/worker/search identity; tighten for corpus caching.
- `apps/common/worker.py::run_parallel`: already completion-order, bounded concurrency. Reuse;
  do not build another thread pool. Journal/resume belongs above it.
- `apps/common/app.py` and managed launcher: config validation, snapshots, run conventions.
- `agents/combat/pv/data.py`: Shard, merge, explicit split membership, state sampling. Existing
  --train-fights becomes a set: repeated IDs cannot implement weighted fight slots. --groups
  logs counts only; it does not balance families. Streaming currently processes shards in turn.
- `agents/combat/pv/train.py`: existing losses/model/optimizer behavior. Extend narrowly or
  extract reusable training support; do not copy a whole trainer to change only the sampler.
- `agents/combat/pv/{recency,train_recency}.py`: fight-indexed replay, hashed manifests and fixed-
  step final export patterns; recency half-life sampling is NOT this protocol's sampler.
- `apps/corpus/`: useful split/allocation ideas, but controller references removed
  `apps.run_rl.combat_loop` / `single_deck` paths and bundles obsolete adaptive/tapering behavior.
  Its deck selector filters MCTS outcomes and costs, contrary to this protocol. Do not revive it
  unchanged or introduce new dependencies on retired modules.
- `apps/continuation/`: native same-root diagnostics, not a corpus workflow; no changes needed.

Build in reviewable increments:
1. Immutable family/provenance manifest and split/eligibility checks; dry-run collection schedule.
2. Pure family/source replay plan with deterministic sampling; owner-local tests before trainer wiring.
3. Minimal trainer integration, fixed final export, sampling diagnostics and checkpoint compatibility test.
4. Corpus orchestration reusing common play/encode/evaluate primitives: stage manifests, complete
   cache identity, per-fight journals, exclusive run lock, idempotent resume and atomic publication.
5. Config, status command, no-game dry run, tiny compatibility smoke, then separately approved launch.

Required tests: original decks excluded from collection but retained in replay; first-round skip;
uniform revisit quotas; split/family leakage; teacher/current/history fallbacks; no outcome-based
sampling; exact slot/state counts; seeded reproducibility; duplicate trajectory rejection; hash/cache
mismatch rejection; resumed jobs do not replay completed fights; partial stages cannot masquerade
as completed checkpoints; monitor/final exclusion; existing fixed-deck tests still pass.

## Local launch implementation notes

Run: `runs/schema=combat_v4/date=2026-10-09/id=champ-corpus-pilot-v1/`.
Entry point: `bash experiments/multi-fight-champ-expert/corpus_run.sh`.
Dashboard: `watch -n 5 'bash experiments/multi-fight-champ-expert/corpus_dashboard.sh'`.

Support-only inventory was sufficient for the proposed quotas. The frozen manifest has 10 original,
40 pilot, 20 development and 100 final card-only families. Exclusion audit preserves the prior
holdout, reconciles later recorded configs/corpus admissions and explicit historical training starts,
and checks exact ten-deck checkpoint ancestry. Some pilot training families may have appeared in
older broad experiments whose weights/data are NOT imported. This is recorded in prepared.json;
no claim that these families were never used anywhere in project history. Card-only grouping is
stricter than the old card+relic fingerprint. Dump source_run_id names a dataset, not a human run;
play UUID and card family are the available grouping evidence.

Preflight budget: 81 tiny (1-simulation) compatibility games on train/development/original families,
2 optimizer steps in an isolated smoke model, plus replay and PyTorch/ONNX parity checks. No final
family is played. No smoke trajectory or smoke checkpoint enters production training. All support
exclusions are static, before gameplay; unexpected smoke failures stop, not outcome-based replacement.

Production collection now fails closed on ANY capped fight: frozen worker does not emit a terminal
trajectory for caps, so silently omitting them would bias replay. Evaluation counts reported caps as
non-wins and shows them separately. Infrastructure errors halt and require explicit review. This
stricter stop gate is an operational safeguard, not a change to search or targets.

Training uses the canonical Shard/model/loss, with selected-row loading to bound RAM. It consumes
three independently sampled, state-shuffled epochs (9,000 optimizer steps), records numpy version,
checks export parity, and publishes only complete model directories. A partial training attempt
requires review before retry; completed fights are journaled and reused under full input identity.

Initial focused verification: 16 owner-local replay/manifest/journal/legacy-workflow tests passed.
Native build completed. Broad CTest: 50/56 passed; failures include retired-import legacy apps and
unrelated native/turn-search checks. These do not establish frozen-worker compatibility; the separate
frozen-worker smoke is required before production launch. See scratch/multi-fight-champ-expert/local-build.log.
