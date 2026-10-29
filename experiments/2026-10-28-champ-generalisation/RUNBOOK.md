# Thousand-deck generalisation — night of 2026-10-28/29

## Outcome sought
Improve unseen-composition Champ performance per wall-clock budget, not training win rate.
One training trajectory; no claim about optimal-play ceilings or training-run variability.

## Frozen experiment
- **1,000 train / 200 development / 200 final** upgrade-agnostic card compositions.
- All50previous NN decks retained unchanged on the training side. Both held-out
  sets exclude the **entire historical corpus training pool**, not just tonight's1,000.
- Human-derived supported A20Champ/no-potion starts. No selection by agent wins.
  Runic Dome/Prismatic Shard and encoder-incompatible cards excluded; Gambling Chip included.
- Inputs: `runs/schema=gumbel_nn_combat_v1/date=2026-10-28/id=champ-1000-inputs-20261028-v2/out`.
  Split SHA256: `caaa140111d5b7e0446bb6dbb30f6f2ca3482993f0a4f786be6ca3dfc7731702`.
  The initial input manifest was superseded before gameplay to tighten the historical-pool guard.
- Warm from fifty-deck **checkpoint40**: weights, optimizer and RNG; no old replay.
- Fixed architecture64/2/2. 5games/deck =5,000/update; ten retained batches;
  no row-address cap. Deck-balanced effective batch2048,500optimizer steps.
- FP32 replay minibatches, BF16 training autocast. Sorted, attention-bounded
  microbatches with weighted gradient accumulation; **one optimizer step per2048draws**.
- Native GPU512/256/32bank,64tokens,9workers×224slots. Full CPU fallback for
  long states, length-bounded CPU batches; nonfinite FP16 predictions get one FP32
  recomputation of the same state. Bad FP32 predictions still fail loudly.
- **20% recorded wall-time validation**, checked at update boundaries, including initial baseline.
  Primary:200unseen-dev decks×10starts=2,000. Separate retention:legacy50+new50,
  ten starts each=1,000. Both count toward validation time. The main dashboard curve
  is **unseen development only**, not a mixture with training decks.
- Final200×20=4,000starts are frozen but never passed to the live loop.

## Simulator correction discovered in qualification
Uncapped Barricade/Entrench produced numeric input95,144.8 and FP16 NaN after1,155/1,200
probe fights. Do not exclude that deck or clip the observation. Player block now
caps at999, including repeated Entrench; overflow-safe increment, regression tested.
Simulator commit `07b7cd4`, preserving earlier Seeing Red/Disarm corrections.
Native process bypasses GPU teardown on a recognised poisoned-CUDA worker exception;
normal teardown now destroys sessions before their borrowed buffers.
**Historical MCTS/PUCT results remain simulator-confounded**; this is not an exhaustive simulator audit.

## Qualification
- 24Python owner/recovery tests passed; native encoding and worker tests passed.
- All1,400starts passed native import/encoding; **no final-test games played**.
- Fixed-rule GPU probe:1,200/1,200train+dev games completed, one explicit episode cap,
  no zero-decision decks. Probe search budget8, not production256.
- Synthetic FP16-overflow stress:1,920GPU predictions successfully recomputed via FP32.
  Separate1024-token CPU-fallback stress passed with bounded batching.
- Real two-process learner failure injection resumed after checkpoint1, preserving
  interrupted output. Final weights **and RNG bitwise identical** to uninterrupted
  two-update control; nonempty optimizer state restored; primary/in-sample validation
  remained separate. Known CUDA retry, STOP override, unknown error and retry-limit tests passed.
- One discarded GPU fit:500×2048draws,1,000train decks only,39,817eligible probe rows,
  1,024,000draws (1,010–1,041/deck),1,000microbatches,39.13s. Peak allocated0.71GiB,
  reserved1.11GiBVRAM; peak hostRSS1.92GiB. **n=1fit profile; variability unknown**.
  No weights saved and no dev rows trained. This is not a full ten-window I/O/RSS bound.
- Artifacts: managed runs `champ-1000-native-preflight-20261028-v2` (Oct28),
  `champ-1000-recovery-qualification-20261028` (Oct28),
  `champ-1000-training-qualification-20261029` (Oct29).
  Logs: `scratch/champ-generalisation-audit/{build-final,tests-final,precision-stress,native-final,training-qualification}.log`.

## Launch and monitoring
```sh
APP=./apps/gumbel_nn_combat_expert_iteration/run.sh
CONFIG=experiments/2026-10-28-champ-generalisation/champ_1000.toml
RUN=runs/schema=gumbel_nn_combat_v1/date=2026-10-29/id=gumbel-nn-champ-1000-20261028-night
nohup "$APP" "$CONFIG" > scratch/champ-1000-launch.log 2>&1 &
"$APP" status "$RUN" --json
"$APP" stop "$RUN" # finish current update and scheduled validation
```
Commit before launch. Confirm actual PID/lock, not stale launcher metadata.
After launch replace **only the verified old web listener on8742**, not training:
```sh
nohup .venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.web \
  "$RUN" --port 8742 > scratch/web-gumbel-nn-champ-1000-20261028-night.log 2>&1 &
```
Verify HTTP response and listener argv. Preserve the other agent's web changes;
only a narrow empty-pair guard was added for disjoint train/validation decks.

Monitor initial validation, first complete update, first post-fit validation, and
full ten-window resource use. Native result telemetry reports explicit `capped`,
`turns`, `steps`, `finished_unix`; log rows include final5%-completion tail seconds.
Collector drains every fight; no short-fight selection or dropping slow tails.
Check unique IDs and5games/train deck/update, all1,000replay groups,500optimizer steps,
1,024,000draws, finite losses, RAM/swap/disk, CPU fallback/precision-retry counters.

Disk: ~20GiB incremental full-window estimate, not a measured guarantee. Before each
collection the loop requires10GiB reserve **plus** at least3GiB or the previous batch's
size, whichever is larger. Clean stop if insufficient. It prunes expired replay and
old inference caches, retains checkpoints/results, and never prunes interrupted archives.
Do not remove unrelated runs or historical data to make room without approval.

## Recovery — bounded, not an endless retry loop
Production app allows **at most2automatic fresh-process retries per run** for recognised
CUDA errors. `out/recovery.jsonl` persists the allowance; `out/recovery.json` gives cause,
PID, checkpoint and stderr path. Learner stderr: `logs/learner-*.stderr.log`.
Interrupted next stage is renamed `.interrupted-<timestamp>-iterNNN`; it is never trained
or appended. Resume latest **committed log-row** checkpoint with optimizer/RNG. Partial
validation is archived and repeated. STOP wins; unknown errors/NaNs/OOM/repeated CUDA
failures stop and require an agent. A live supervisor's brief cooldown is `recovering`.

For a later explicit restart: inspect original stderr + archived native worker.log,
GPU health, actual loop lock and orphan worker argv first. Do not overlap workers.
Remove STOP only if intentionally continuing. Then:
```sh
nohup .venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.supervisor "$RUN" \
  > "$RUN/logs/manual-supervisor.log" 2>&1 &
```
Budget exhausted: inspect/escalate; do not repeatedly increase `--retries`. Do not change
search/architecture silently or treat infrastructure errors as losses. Original launcher
metadata can remain failed after a manual restart; lock/process liveness is authoritative.

## Analysis / final test protocol
Primary point estimate is unseen-dev macro win rate. JSON exposes family-clustered
approximate95% intervals and **paired family-mean improvement versus initial**. Each
family is one cluster across its fixed starts. Conditional seed/Wilson statistics remain
available, but are not unseen-family uncertainty. Intervals are unadjusted for repeated
looks; they do not measure variability across training runs.

Choose the checkpoint with highest committed unseen-dev macro win rate; ties choose
the earlier checkpoint. Freeze its identity/hash **before** final evaluation. Evaluate
that model and checkpoint0 on the same4,000reserved final starts using the frozen worker,
zero Gumbel noise and production search. Never train those outputs or use final outcomes
to choose another checkpoint. Report paired family-clustered95% improvement,200families,
4,000starts, and one training run. Final source: `inputs/FINAL_TEST_DO_NOT_USE_FOR_SELECTION.jsonl`.

Canonical historical references are MCTS20k and **PUCT D5**, kept distinct from ten-deck
PUCTu6/other policies. Dev has321matching full-start/seed historical trials, only a subset
of the2,000primary trials. Input provenance records both source runs. These are context,
not same-rules controlled comparisons or optimal-play ceilings. No inference of poor
play/headroom from a low absolute win rate. JSON-only historical outputs are not exhaustively inventoried.

## Live state
Launched **2026-10-29 07:57UTC** from project commit `0575496`, simulator `07b7cd4`.
The earlier session stalled after qualification; no overnight thousand-deck run occurred.
Managed launcher PID674601, supervisor/app674625, learner674659 (initial PIDs; retries change them).
Warm checkpoint40 verified: source SHA256 `d0993c86d02ca70d5fa6f9e047c04575a41f041d8ea7f4174dce98ecebd4e3ae`.
Checkpoint0 committed; native worker active on the initial2,000-start development baseline,
then1,000retention starts before first5,000-game collection/update. First full live update
and full-window resource use are not yet verified.
Port8742 verified old fifty-deck web PID536851 and replaced only that listener;
new web PID674878, HTTP200. Active pointer updated; previous pointer preserved as
`scratch/gumbel-nn-active-run-before-1000.json`.

