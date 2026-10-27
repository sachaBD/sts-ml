# Useful inference throughput investigation — runbook

## Goal / authorization

User authorizes autonomous investigation and implementation to maximize useful inference throughput on this Ryzen 3600X (6 cores / 12 logical CPUs) + RTX 3060. Specifically investigate variable-length-game tails. No new model training. Do not drop/cap additional long games, change search budget, truncate states, or silently bias collection for benchmark speed.

## State at start

Working tree already contains this conversation's topology, checkpoint-resume, timing, dashboard and test changes; preserve them. Nothing committed by this investigation yet. Completed training run: `runs/schema=combat_v4/date=2026-10-25/id=nn-gumbel64-panache-128x4-v1`.

Use **iteration 10** frozen checkpoint (128 wide / 4 layers / 4 heads), 64 simulations, rollout mix .5, 8 particles, Gumbel scale 1, 10 CPU workers. Same 2,000 iteration-10 starts for all matched comparisons. Full game completion required.

## Planned sequence

1. Establish reproducible benchmark launcher, progress/telemetry, resumable result artifacts. Save baseline executable before changing native code.
2. Measure baseline 1,024 and fixed smaller batches with identical frozen model/starts. Separate evaluator setup, gameplay, real evaluations, padded work, CPU use, and time spent completing final fractions of games. Capture `nvidia-smi` telemetry during gameplay (not training/build).
3. Follow evidence: investigate ready-first scheduling / more independently scheduled groups and shape-adaptive GPU inference for the draining tail. Keep configuration explicit and preserve default behavior until verified.
4. Validate numerical/model I/O behavior and game completion; benchmark shortlisted configurations with repetitions, alternating order. Report sample sizes/ranges, not unsupported confidence intervals.
5. Write `THROUGHPUT_REPORT.md` with recommended settings, measured results, trade-offs, correctness checks, remaining limitations. Update this runbook throughout.

## Artifact conventions / recovery

- Benchmark source: `experiments/25-10-26/a-gumbel/nn/throughput.py` (to be created).
- Generated artifacts: `runs/nn-throughput-20261025/`.
- Logs: `runs/nn-throughput-20261025/logs/`; benchmark `report.json` and per-trial `summary.json` are completion markers. Interrupted trials must be archived or assigned fresh names, never appended to sample files.
- Builds remain under `build/nn/`; build logs in the artifact tree. Python: `.venv/bin/python`, dependencies unchanged.
- Long commands run in background after chat notice, output redirected to named logs.
- On recovery, first read this file + report if present; inspect git diff, process list (`nn_play`, `throughput.py`) and artifact summaries. Do not start a duplicate GPU job.

## Baseline prior evidence (not matched benchmarking)

Nine 2,000-game training updates had 50–61% batch fill, 84% evaluator-busy wall time; mean useful rows 56%. `/usr/bin/time` for resumed run averaged 2.47 logical CPUs. Evaluator busy includes transfers and CPU handling and is NOT GPU compute-capacity utilization. Different weights/game lengths across updates confound those throughput comparisons.

## Progress

- Runbook created; native evaluator and worker inspected. Worker currently alternates two groups and blocks on each group's future in fixed order. Game slots are replenished only while starts remain; 2,000 starts = 2,000 slots leaves no reserve. Evaluator batches FIFO whole requests and always computes a fixed 1,024 rows, including padding. Tail inference is a plausible waste source but must be measured.

### Fixed-shape first screen completed (~16:10)

Command (completed; safe to rerun, completed summaries skipped):
`PYTHONPATH=. .venv/bin/python -u experiments/25-10-26/a-gumbel/nn/throughput.py --binary runs/nn-throughput-20261025/nn_play-before-investigation --batches 1024,512,256 --tag fixed-screen`

Results in `runs/nn-throughput-20261025/fixed-screen-report.json`. Single trials: 1024 = 52.8811 s gameplay / 58.1k useful eval/s / last 10% 13.52 s; 512 = 40.6374 s / 75.8k / last 10% 8.79 s; 256 = 41.9284 s / 73.4k / last 10% 5.84 s. Leaf totals estimated from rounded mean in old executable; new instrumentation will provide exact counts. FP16 shape changes yielded 833 vs 832 wins, so game paths are not guaranteed identical across shapes; do not claim bit-exact equivalence. New numerical validation pending.

### Implementation in progress

- `gpu.cpp`: optional smaller-shape engine bank; chooses smallest shape fitting gathered real rows. One evaluator thread, shared fixed-address buffers, one captured graph/binding per engine. Existing fixed-shape default retained. Exact periodic `gpu_stats` adds useful leaves, padded capacity, CPU fallback count, per-shape runs.
- `play.cpp`: explicit `--groups` and `--scheduler ready`; ready-first polling avoids waiting on a particular group's future. Default ordered/two-group behavior retained.
- `net.hpp`: optional bank parameter; `play.cpp --gpu-bank comma,separated,onnx,paths`.
- Build launched as background job 8; log `runs/nn-throughput-20261025/logs/build.log`. Check this before launching new-native benchmarks. Baseline binary remains preserved separately.
- Next: build, CPU scheduler parity, GPU numerical checks, screen adaptive512 bank256/128/32 and ready-first groups4/8, then matched repeated shortlist trials. Benchmark script accepts `--adaptive`, `--groups`, `--scheduler`; use unique `--tag` for different engine banks (bank not encoded in trial name).

### Validation passed; adaptive screen running (~21:37 machine clock)

The host wall clock jumped between early screen and validation (possibly suspend/resume); use monotonic trial durations and repeat shortlisted comparisons interleaved before conclusions.

`verification/report.json`: old CPU / refactored ordered / ready2 / ready4 all matched HP, wins and decisions on 8 fights at 4 sims. 128 recorded states checked against CPU ONNX for GPU shapes 32/128/256/512/1024: 100% top-policy-action agreement; maximum value absolute error .00194 and maximum legal-logit absolute error .0205. These are sampled checks, not exhaustive equivalence.

Native build succeeded; `build-wait.log` includes optional `--gpu-wait-us` implementation. Default 1000 us preserves stall-wait behavior; 0 dispatches available work without intentionally waiting (still condition-variable sleeps when empty, not spinning).

Background job 11 is running sequentially, log `runs/nn-throughput-20261025/logs/adaptive-screen.log`:
1. `throughput.py --batches 1024,512 --tag refactor-fixed`
2. `--batches 512 --adaptive 256,128,32 --tag adaptive-screen` (ordered2)
3. same + `--scheduler ready --groups 4`
4. same + `--scheduler ready --groups 8`
5. ready4 + `--wait-us 0`
6. ready4 + `--wait-us 100`
7. `--batches 1024 --adaptive 512,256,128,32 --scheduler ready --groups 4 --wait-us 0 --tag adaptive-screen`

Trial names now include adaptive bank and nondefault wait to avoid collisions. Each has summary.json completion marker. The per-tag aggregate report is overwritten by each invocation; use per-trial summaries as the durable source of truth.

### Additional checks / caveats

- Python owner tests (`test_model`, `test_loop`) still pass (4 tests).
- Root-state training token lengths (50,501 stored states) average 31.87, max39, 56.2% <=32. This is NOT the searched-leaf distribution; token bucketing may be a later opportunity, not yet implemented.
- Same-shape legacy/refactored GPU full-game runs differ on 33/2,000 wins and 142/2,000 decision counts, while total useful leaves differ by <1%. Small fp16/search-threshold sensitivity and scheduling-dependent row arrangements are possible; do not equate output bitwise equivalence with statistical play-strength preservation. CPU scheduler tests are exact; GPU numerical checks are tolerance-based. No new play-strength claim from these throughput trials.
- Adaptive-bank setup is warm-cache here. Before recommending it for short training iterations, measure cold per-checkpoint engine build/export overhead; a faster gameplay phase could still lose full-update time. One-shape batch512 is already an attractive minimal change.

### Adaptive / follow-up screens completed (~21:47)

Job 11 completed; durable summaries under `trials/adaptive-screen*` and `trials/refactor-fixed*`. Job 12 completed; follow-up summaries + `cold-fixed/`, `cold-bank/` nested artifact roots. Use `analyse_throughput.py --tag NAME` for readable tables and telemetry JSON.

Findings (single screens, not final replicated claims):
- Instrumented default 1024 ordered2: 48.33 s / 63.7k eval/s, tail10% 11.63 s.
- Fixed512 ordered2: 37.37 s / 82.5k; adaptive512 bank256/128/32 ordered2 wait1000: 33.99 s / 90.7k.
- Ready4 / ready8 with same bank+wait1000: ~91k, but consume ~5.7–5.9 logical cores vs ordered2 ~4.3. Not a meaningful throughput gain for the added CPU.
- Adaptive512 wait100: ready4 31.59 s / 97.5k, ordered2 32.13 s / 95.9k. Wait0 similar. Ready4 max1024 slower (~89.4k).
- Network-only copied-input/output reference: batch512 median132.8k eval/s, batch1024 140.1k (3 blocks x50 calls). This is not gameplay throughput. Shape128 has worse latency than256 with these engines; try omitting128 before shipping extra models.
- Cold-fixed512: export/preparation3.78 s + evaluator setup4.01 s + gameplay37.71 s. Cold bank512/256/128/32 wait100: prep9.02 s + setup7.91 s + gameplay31.24 s. Extra models can negate gameplay savings on 2k iterations. These are one fresh cache-backed build each; timing variability unmeasured.

### Current implementation / next

`loop.py` now accepts opt-in `--gpu-bank 256,128,32`, `--gpu-wait-us`, `--parallel`, `--groups`, `--scheduler`; automatically exports matching same-weight bank models. Defaults remain unchanged. Resume rejects changing previously fixed inference settings; new topology/weight/layout contracts preserved. Owner tests: 7 pass. Dashboard now uses actual summed engine capacity for adaptive fill rather than max batch size.

Added detailed evaluator phase timing and queue/packing stats (host wall times, not CUDA occupancy) after screens. Build job13 running; log `logs/build-profile.log`. Previous screening native binary saved at `nn_play-screen-version`.

Next: screen reduced banks512/256/32 and512/32; then interleaved repeats of baseline1024, simple512, chosen adaptive setting. Measure4k jobs to quantify reserve benefit; a short cold reduced-bank trial to judge full-iteration tradeoff. Finish report + runbook with exact commands, raw artifact paths and no training launch.

### Reduced bank + profiling (~21:52)

Reduced banks completed (`reduced-bank*`): 512/256/32 ordered2 wait100 =31.67s /97.3k eval/s; 512/32 =33.13s /93.0k. Prefer the former as max-throughput candidate; one fewer engine than original bank. Detailed profile for 512/256/32: host evaluator inference23.47s, concatenation2.17s, H2D1.77s, D2H1.31s, scatter/fulfillment1.63s, ~1.3s outside evaluator. Worker packing summed12.48s across workers (overlapped, not additive to gameplay wall). Queue wait mean ~7.5ms per real leaf. These are host timing categories, not CUDA occupancy.

Native instrumentation now also counts searched-leaf token lengths (sum/max, >32 and >40). Build15 succeeded (`logs/build-tokenstats.log`). Job16 running `token-screen` baseline48 and token40 with batch512 bank256/32 ordered2 wait100; log `logs/token-screen.log`. Benchmark `--tokens` controls all bank shapes; CPU fallback preserves oversized states. Do not recommend token40 until its fallback count and numerical parity are checked. Loop already supports configurable gpu-tokens.

Next: numerical check40 if useful; repeated interleaved confirmation of default1024 vs simple512 vs selected adaptive configuration (3 trials/config, same frozen checkpoint/starts, no concurrent workload), plus4k reserve experiment and cold reduced-bank overhead. Default pipeline settings remain unchanged; all improvements opt-in.

### Final confirmation running (~22:00)

Token screen complete: all ~3.08M searched leaves fit39 tokens, CPU fallback0; mean31.74, ~45% >32. Nevertheless token40 was substantially WORSE (49.51s vs31.56s at48), with inference42.29s vs23.42s. This is an engine/tactic/shape performance finding, not a reason to truncate. Reject token40; keep48. Do not spend further time on numerical40 checks for the rejected candidate.

Job17 running, log `logs/confirmation.log`. One frozen instrumented native binary throughout. Three repeats/config interleaved in order A/B/C, C/B/A, B/A/C:
- A: `throughput.py --batches 1024 --tag confirm --repeat-offset R`
- B: `--batches 512 --tag confirm --repeat-offset R`
- C: `--batches 512 --adaptive 256,32 --wait-us 100 --tag confirm --repeat-offset R`
with R=0,1,2. Each uses2,000 starts, ordered2,10workers,200slots/worker,48tokens,64sims,rollout.5,Gumbel1. Completed markers skip on rerun.

After repeats, same background job runs:
- `throughput.py --batches 1024 --games 4000 --tag reserve`
- `throughput.py --batches 512 --adaptive 256,32 --wait-us 100 --games 4000 --tag reserve`
- `throughput.py --out runs/nn-throughput-20261025/cold-reduced --batches 512 --adaptive 256,32 --wait-us 100 --tag cold-reduced`
- `analyse_throughput.py --tag confirm` and `--tag reserve`.

4k starts = same2k prefix +2k fresh deterministic seeds, NOT duplicate fights. These reserve tests are single exploratory trials; full confirmed comparisons use exactly matching2k starts. Avoid CPU/GPU benchmarking/tests concurrently with job17. Current useful stats include detailed host timings, queue latency and token counts. CPU callbacks / fallback remain full-input exact-capability paths. No model training launched.

Final remaining work after job17: verify exit0 and complete trial markers; inspect analysis-confirm.json, telemetry counts and model/start/binary hashes; run owner CTest tests after benchmarking; write THROUGHPUT_REPORT.md including repeat ranges, cold build/export costs, CPU/GPU utilization interpretation, rejected approaches, model/deck scope, FP16 caveats, exact recommended command and further headroom. Update readme, test CLI/pipeline bank options, leave defaults backward compatible. Run git diff --check. Mark this runbook complete and final report location.

## PAUSED BY USER — host outage / new real-time session
User requested stop work and discussion. Investigation process tree rooted at PID257699 paused with SIGSTOP if still alive; do not resume or launch anything without renewed direction. No handoff commit made yet. Host clock jumped during confirmation; confirm-r2 baseline /usr/bin/time reports34h despite~62s monotonic gameplay. Treat that trial as outage-contaminated and do not claim clean three-repeat confirmation. Raw artifacts retained.

## Handoff after design discussion / implementation

The user approved the worker-local queue design and requested implementation. This supersedes further autonomous tuning of the old group scheduler. New implementation and verification live in `agents/combat/nn/WORKERS.md`; report in `THROUGHPUT_REPORT.md` distinguishes old batching evidence from new scheduling correctness. No background benchmark or training is left running.

Outage caveat is mandatory: exclude `confirm-b1024-...-r2` and use the first TWO paired confirmation repetitions (r0/r1) for conservative old-scheduler comparisons. GNU-time CPU aggregates for the disrupted trial are invalid. The raw unfiltered analysis utility does not automatically recognize host-outage contamination. New local scheduler is opt-in, not claimed faster, and needs review before a fresh controlled throughput comparison.

Local implementation checks: native/Python owner tests; real8-fight CPU parity and32-fight GPU sample-integrity smoke; ASan/UBSan pass. TSan unavailable due host memory-mapping failure. Source includes bounded queues, immutable input lifetime, per-request generations, progress/request timeouts and a supervised fatal shutdown for native hangs. See WORKERS.md for exact flags, commands and remaining limitations.
