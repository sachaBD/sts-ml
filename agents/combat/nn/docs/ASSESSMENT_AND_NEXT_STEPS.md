# Pipeline assessment and next steps

Decision summary from the 27 October 2026 throughput discussion. Read alongside the [GPU measurements](GPU_PERFORMANCE.md), [benchmark instructions](INFERENCE_BENCHMARK.md), and historical experiment reports. These are workload-specific engineering judgments, not training-strength results.

## Bottom line

**Stop open-ended hardware tuning and return to a bounded training/strength experiment.** The current GPU configuration is reasonable for this network and machine. We did not prove it optimal, and we did not establish that this transformer should be the long-term architecture.

Prioritize independently reviewing the CPU-overhead improvements and measuring learning progress. Idle CPU time is not itself a defect; simulator workers waiting for a shared evaluator can be expected.

## Decisions and evidence

| Question | Assessment / decision | Evidence and qualification |
|---|---|---|
| Is the worker/coordinator split fundamentally wrong? | No evidence warrants replacing it. Retain worker-local simulator/search ownership. | Each slot has one outstanding NN request; the coordinator owns batching/inference. Ownership and cancellation contracts are in [WORKERS.md](../WORKERS.md). No new controlled scheduler superiority claim. |
| Did local scheduling clearly beat legacy scheduling? | Not established. | Historical legacy bank runs reached 91.6–95.5k real NN evaluations/s (n=2); worker-packed local reached 91.7k (n=1). These were not a fully controlled paired scheduler comparison. See the [historical investigation](../../../../experiments/25-10-26/a-gumbel/nn/THROUGHPUT_REPORT.md). |
| Was worker-side input packing worthwhile? | Yes, evidence of a substantial operational gain. | Matched one-update comparison: 64.7k → 91.7k NN evaluations/s; full-update games/hour improved about 19%. n=1/version; no strength claim. [Packing report](../../../../experiments/25-10-26/a-gumbel/nn/PACKING_REPORT.md). |
| Retain experimental GPU overlap? | No. Removed during cleanup; retain stable synchronous inference. | The experimental run measured 86.7k versus 91.7k evaluations/s without overlap, and lower row fill (85.8% versus 91.2%). n=1/version, not definitive attribution, but no evidenced gain. Raw artifact: `runs/nn-pipeline-bench-20261027/`. Recoverable source patch: `scratch/nn-cleanup-20261027/gpu-overlap.patch` (base commit recorded alongside it). |
| Increase GPU batch to about 5k? | No. Keep maximum 512 as the practical default. | Two short trials/shape: 5120 added only about 3.6% prepacked throughput versus 512, with about 9.7× batch latency. 2048's CPU-fed advantage was about 5.8%, not a demonstrated self-play gain. Current 2016 active slots cannot fill 5120 requests. |
| Reduce padded GPU context to 40? | Keep 48 with the available engines. | Two trials/shape: cached 40-token engine was about 41% slower. This does not prove 40 is intrinsically slower: different tactic/build histories were not isolated. A fresh 40-token build would be the cheap follow-up if revisiting. Never truncate long states. |
| Use idle CPU threads for additional inference? | Not worth hybrid routing complexity now; retain required CPU fallback. | CPU-only tests: about 3.3k evaluations/s on two callers, 5.8k on four, best tested batch 8. Nominal addition is only a few percent against 92–131k GPU-driven/backend rates, before simulator/core/cache contention. No actual hybrid speedup measured. |
| Custom kernels or architecture-specific fusion work? | Defer. | About 3.58ms summed kernel time plausibly accounts for most inference work. Roughly 49% dense matrix/vector kernels, 11% attention, 24% normalization/residual, 16% layout/embedding-related kernels. Semantics inferred from symbols; profiled n=3 graphs, not independent trials. No simple missing setting was identified. |
| Is GPU busy percentage the optimization objective? | No. | NVML busy measures active time, not useful compute efficiency. Host synchronization time is not kernel-active time. Prefer measured real throughput, complete update time, and ultimately strength improvement/hour. |
| Is the arithmetic ceiling a reasonable throughput promise? | No. Withdraw any implied 200–250k expectation. | About 81M FLOPs/evaluation and historical large-GEMM 28TFLOP/s imply a compute-only reference of about 346k evaluations/s. The different workload omits substantial runtime costs; neither that rate nor 200–250k is established attainable. |

Batch/token/CPU measurements, ranges, sample sizes, execution paths and raw artifact locations are in [GPU_PERFORMANCE.md](GPU_PERFORMANCE.md). Short-run differences are not confidence intervals or training-strength evidence.

## CPU-overhead work: promising, not yet independently accepted

The separate CPU-only workstream reported warm replay loading **3.4 → 0.68 seconds** and warm export **7.8 → 0.064 seconds**. At the time of this discussion these were the other agent's reported results, not independently reviewed integration results. They must not be advertised as an established sustained learning-loop speedup.

Before adopting them, review changed-weight export correctness, cold initialization/fallback costs, replay equality/eviction, memory growth, resume compatibility, and production-loop integration. Keep their status distinct from the GPU sweep findings. The [overhead specification](../OVERHEAD_SPEC.md) defines acceptance requirements.

Persistent native inference services and TensorRT weight refitting remain deferred. A live process alone does not update model weights; every GPU shape and the CPU fallback must use the same version. They were not implemented or validated in this assessment.

## Suggested training starting configuration

Use the reviewed worker-packed path, not the experimental overlap option. The discussion's starting point was:

```text
--scheduler local --workers 9 --parallel 224
--gpu-batch 512 --gpu-tokens 48 --gpu-bank 256,32 --gpu-wait-us 100
```

These are Python-loop arguments, not a complete launch command. Native `nn_play` uses `--threads` instead of `--workers`. The experimental `--gpu-pipeline` option was removed; do not pass it. The maximum batch is 512; the smaller 256/32 engines serve partial batches, especially the collection tail. Preserve full-input CPU fallback for states exceeding the token limit. Worker count/slot count were not newly optimized in this discussion. This records a recommendation, not a change to CLI defaults or authorization to launch a run.

## Before a long training run

1. **Review the exact code/settings being launched.** The experimental overlap changes were archived and removed during cleanup. Do not conflate a successful inference benchmark with verification of later integrations. Run the relevant correctness tests and a short real-game collection/checkpoint smoke test after integration.
2. **Review the overhead work before enabling it.** Keep new options off until its equivalence/resume behavior is accepted. Do not silently combine unreviewed changes with the first long learning run.
3. **Start bounded.** Run a short play → train → checkpoint/export cycle with fixed held-out evaluation fights/settings. Verify sample/result counts, finite normalized policies, readable compatible checkpoints, and complete updates. Do not append to interrupted collection stages; preserve quarantine/resume rules.
4. **Measure the complete loop across multiple updates.** Record collection, engine setup, replay loading, training, save/export and evaluation overhead. A one-update warm-cache benchmark does not include building newly exported engines for the next update. No sustained post-overhead baseline was established here.
5. **Use held-out strength checks.** Confirm progress before extending the run. Loss reduction and row throughput alone do not establish a stronger autonomous agent. Choose an explicit initial training budget/evaluation cadence; this discussion did not select a universal recipe or prove one.

No training was launched as part of the new GPU/CPU inference measurements in this discussion.

## Cleanup verification

The experimental overlap diff was saved to `scratch/nn-cleanup-20261027/gpu-overlap.patch` before restoring the eight affected production/experiment files to base commit `80aa3bb`. Patch reapplication was checked with `git apply --check`. Full tracked/untracked pre-cleanup snapshots are preserved alongside it; these are local ignored recovery artifacts, not portable committed source.

The stable `nn_play`, `nn_bench_inference`, `nn_workers_test` and `nn_packing_test` targets rebuilt successfully. All six selected CTest entries passed (four owner Python test entries plus native worker/packing tests). The real-model integration cases are opt-in and were skipped, not newly verified. Benchmark CLI usage/invalid-deadline checks and a real-GPU 1-second-per-mode smoke run passed. That final smoke measured about 109k prepared / 102k CPU-fed evaluations/s, lower than the earlier settings sweeps; it was not a controlled performance comparison, and the cause of the difference was not established. No claim of a fresh sustained throughput baseline follows from cleanup. No training or new real-game collection was launched.

## What to optimize and how to count it

Keep learning semantics fixed when evaluating infrastructure changes. Count **fresh searched-decision rows per full-loop hour**, not NN leaves or rolling replay size. In the worker-packing snapshot, about 3.08M NN evaluations produced about 52.9k new searched-decision rows; the roughly 255.6k reported states were the multi-iteration replay, not newly generated data. About 58 NN evaluations per new row reflects this search setting, not a universal conversion.

Even fresh rows/hour is only an operational proxy: longer fights can generate more rows without better learning. The ultimate objective is **held-out playing-strength improvement per hour**, with uncertainty and evaluation sample size reported. The earlier proposed 40–45s full-update goal was a planning hypothesis, not a validated attainable target or an acceptance gate before starting training.

Reopen performance work if complete-loop measurements reveal a new dominant bottleneck, the model changes substantially, or a specific cheap setting is implicated. Do not delay learning experiments merely to make CPU/GPU utilization percentages look full.
