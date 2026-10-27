# Throughput investigation — handoff

## Status / scope

The original batching investigation was interrupted by a real host outage. The user then approved a simpler **worker-local scheduler**, implemented separately in `agents/combat/nn/workers.{hpp,cpp}`. Its design, review entry points, failure behavior and verification are documented in `agents/combat/nn/WORKERS.md`.

**Do not attribute the old batching measurements below to the new scheduler.** New local scheduling has correctness smoke coverage, not a validated throughput improvement or training-strength result. Legacy defaults remain selectable/unchanged for review and comparison. No extra neural-network training was performed during throughput investigation or worker implementation.

## Findings from the legacy scheduler

Frozen iteration-10 panache network (width128/layers4/heads4), identical2,000 starts, Gumbel64, rollout.5,10 Sim threads,200 slots/thread,48 tokens, all games finished. The first TWO paired confirmation repetitions (r0/r1) completed before the outage. Their observed ranges, not confidence intervals:

| Legacy configuration | Gameplay seconds | Useful GPU evaluations/s | Final10% completion seconds |
|---|---:|---:|---:|
| fixed1024, wait1000us |49.75–50.28|61.3–61.8k|12.69–13.05|
| fixed512, wait1000us |37.06–40.04|77.0–83.2k|7.59–8.16|
| adaptive512/256/32, wait100us |32.26–33.65|91.6–95.5k|2.61–2.89|

n=2 sequential interleaved trials/configuration. Adaptive row fill ~88%, versus ~56% for fixed1024. Roughly1.5x useful throughput in this specific frozen-model workload; not a universal hardware estimate. Inference output paths can differ due to FP16 numerical sensitivity: these are matched starts/settings, not guaranteed bit-identical trajectories.

The third baseline trial (`confirm-b1024-...-r2`) is **outage-contaminated**: GNU time reports34h elapsed, versus~62s native monotonic gameplay. Exclude it. Earlier chat described clean three-trial confirmation too strongly. Do not use unfiltered `analysis-confirm.json` CPU/range aggregates as final evidence. Raw per-trial files are authoritative; use r0/r1 for the table above.

Other exploratory screens (usually n=1/configuration):
- Ready-first4/8 groups increased CPU activity without enough useful throughput improvement to justify adopting them.
-40-token engines were slower than48 despite zero fallback and less padding (49.5s versus31.6s); reject. Fewer padded elements does not imply faster TensorRT tactics.
- Larger4k collections sustained occupancy better, but change games/update. Reserve trials straddled host interruption; treat as exploratory, not confirmed.
- Extra engine export/build overhead can negate short-update gameplay savings. One cold fixed512 trial: preparation3.78s + setup4.01s + gameplay37.71s. One cold reduced-bank trial:7.30s +7.01s +36.10s. These are single cache-backed builds under different conditions, not a fair replicated end-to-end win for the bank.
- Network-only full-batch512 reference ~133k eval/s (3 blocks ×50 calls). Not simulator throughput, FLOP utilization, or a universal upper bound.

## New worker-local implementation

- N SimWorkers own K slots each; no simulator/search migration.
- Shared atomic allocator for fresh starts; per-worker bounded request/completion deques with ordinary mutexes.
- One coordinator gathers fairly, calls the existing backend synchronously, and routes predictions by slot/request generation.
- GPU backend does not start its old dispatch thread in local mode. Model/token/sample schemas remain compatible.
- Waiting input views remain immutable. Cancellation retains owners until coordinator completion; native hangs fail the process after explicit shutdown grace, never become game losses.
- CPU/GPU correctness checks and native fault-injection tests are recorded in `WORKERS.md`. Main loops are `SimWorker::run` and `Coordinator::run`.

Opt-in settings for review (not a training-launch approval):

```sh
--scheduler local --workers 9 --parallel 224 \
--gpu-batch 512 --gpu-tokens 48 --gpu-bank 256,32 --gpu-wait-us 100
```

For native `nn_play`, use `--threads` instead of `--workers`. Slots bound active games to9×224=2,016. Batch shape, slot count and worker count are independent knobs.

## Next session

1. Review the worker code and ownership/timeout contract before making it the default.
2. On a stable host, compare local vs legacy using the SAME frozen weights, starts, inference shapes and CPU-thread budget. Report exact real eval/s, all-game completion time, tail time, memory and startup overhead. Do not use CPU activity as the objective.
3. Coordinator now packs inputs synchronously; that is intentionally simple but may serialize packing with inference. Measure before optimizing. Existing profiling can separate packing/inference/transfers.
4. Keep short-update cold engine/export cost separate from warm inference throughput. No training-strength conclusion follows from batching tests.

## Artifacts / recovery

- Detailed chronology/commands: `THROUGHPUT_RUNBOOK.md`.
- Frozen-model raw benchmark artifacts: `runs/nn-throughput-20261025/` (local ignored generated outputs).
- Every completed trial has `summary.json`, `command.json` hashes, full results/samples, monotonic progress, worker/time logs and GPU telemetry. Interrupted directories must be archived, never appended.
- Worker implementation verification: `scratch/nn-local-workers/`.
- Source utilities: throughput.py, analyse_throughput.py, verify_throughput.py, peak_throughput.py beside this report. The benchmark CLI accepts `--scheduler local`. For equal application-thread budgets compare legacy9 SimWorkers + its GPU thread against local9 SimWorkers + coordinator, with matching slot counts and all other settings.
