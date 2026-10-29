> Historical measurements of the pre-infra implementation. Current backends and
> component benchmarks are in `infra/`; the old native benchmark was removed.

# GPU inference: measured settings and interpretation

## Recommendation

For the tested panache network on this RTX 3060, retain **batch 512 and 48 padded tokens**. Larger batches offer little backend throughput improvement. The available 40-token engine is substantially slower, despite less theoretical work. Do not interpret this as a universal optimum or a reason to commit to this model architecture.

Further work should be limited to inexpensive backend/configuration checks unless a larger optimization effort is explicitly warranted. In particular, the arithmetic-only ceiling below is not a throughput target or a promise of recoverable performance.

## Scope and reproduction

Hardware: Ryzen 5 3600X (6 cores / 12 threads), RTX 3060 12 GB, WSL2, Windows NVIDIA driver 572.61. Model: width 128, four layers, four attention heads, feed-forward width 512; same trained weights throughout the settings sweeps.

The [native benchmark](INFERENCE_BENCHMARK.md) calls the production synchronous `gpu_evaluator`: ONNX Runtime/TensorRT, FP16 input/compute enabled, optimization level 3, CUDA graphs, pinned-buffer concatenation, transfers, result scatter and validation. The experimental staged pipeline was not used in these measurements and was later removed during cleanup. It measures the deployed backend, not just GPU arithmetic. No simulator, search, training or lifecycle overhead is included in timed throughput.

`prepared` repeatedly uses a prepacked batch. `cpu_fed` uses one producer thread generating and FP16-packing fresh synthetic inputs through bounded reusable buffers. Inputs have 32 populated tokens, valid embedding-index ranges, four action tokens, and padding to the engine's fixed token length. They are not a representative distribution of real combat states. All batches were full. Setup and warm-up were excluded from throughput timing.

All reported trial ranges are observed min/max, **not confidence intervals**. Results are short engineering screens, not playing-strength or sustained learning-loop measurements.

## Batch-size sweep

Two invocations per shape; each mode timed for 1 second with 3 warm-up calls. Orders 256/512/1024/2048/5120, then reverse. 48 padded tokens throughout.

| Batch | Prepacked evaluations/s | CPU-fed evaluations/s | Prepacked batch cycle ms |
|---|---:|---:|---:|
|256|122.4–122.7k|115.4–117.4k|2.086–2.092|
|512|130.6–131.6k|119.6–120.3k|3.889–3.920|
|1024|130.5–130.8k|121.5–122.1k|7.827–7.849|
|2048|131.4–132.6k|125.9–127.8k|15.447–15.585|
|5120|135.6–136.1k|122.4–123.4k|37.629–37.772|

Mean prepacked throughput rises only 3.6% from 512 to 5120, while cycle latency rises 9.7×. CPU-fed throughput is highest at 2048 (+5.8% versus 512), but this small difference is not an established integrated self-play gain. The current local-worker configuration has about 2016 active games, so it cannot supply 5120 simultaneous leaves without increasing slots.

Sampled device-wide warm-process peak VRAM: 617/649/693/781/1101 MiB respectively. Samples every 200 ms include startup and both modes and can miss short peaks; these are not isolated inference allocations. New 2048/5120 engines took 16.3/17.0 s to prepare; subsequent cached setup ~0.6 s. Every individual invocation completed in 2.9–19.8 s. The ten invocations totaled 62.8 s including engine preparation.

Existing engines were reused for 256/512/1024; new engines built for 2048/5120. Different tactic/build history is part of this deployed-settings comparison, not experimentally isolated from batch size.

Artifacts: `scratch/nn-inference-bench/batch-sweep/{results.json,run.py,prepare.py}` and per-shape commands, logs, models and telemetry. Source checkpoints were checked for matching specs and exact equality of all state entries before reuse/export.

## Token-length check: why less padding was slower

Batch 512; two invocations per shape, each mode timed for 2 seconds with 3 warm-up calls. Order 48/40, then 40/48. Same 32 populated tokens and seeded input generation.

| Padded tokens | Prepacked evaluations/s | CPU-fed evaluations/s | Prepacked batch cycle ms |
|---|---:|---:|---:|
|48|125.0–128.3k|117.2–118.5k|3.990–4.096|
|40|74.0–74.7k|71.9–72.4k|6.854–6.917|

40-token prepacked throughput was 41% lower on average. All four invocations completed in about 20 seconds total; each took ~5 seconds. Cached models/engines were reused, not rebuilt. Initializer inspection found 84 common named initializers: 76 identical arrays, with the remaining differences restricted to shape constants and zero-mask shape changes rather than alternative trained weights.

### What the result does and does not mean

A TensorRT engine is a compiled execution plan, not a device performing a fixed cost per mathematical operation. Each fixed shape can select different kernels, tile sizes, layouts and fusion opportunities. Throughput can therefore change discontinuously as a shape changes.

If execution efficiency stayed constant, fewer padded tokens would reduce work: linear projections scale roughly with token count, and attention matrix work roughly with its square. 40 instead of 48 would reduce those counts by 17% and 31% respectively. **That assumption of unchanged execution efficiency is not guaranteed.**

For example, a kernel processing 16-token tiles might cover both 40 and 48 positions with three tiles, leaving no arithmetic benefit internally, while a shape-specific dispatch chooses a different implementation. 48 is divisible by 16 and 40 is not. This is an illustration, **not evidence that a particular 16-token tile or alignment caused our slowdown**. Fusion eligibility, layout conversions and tactic selection are other hypotheses.

Both shapes use the same tensor schema, matching weights, seeded populated data, requested precision and backend settings. The observation is robust across two short trials, but the engines have different build histories. We did not profile the 40-token kernel sequence or rebuild it under the same current conditions. Therefore:

- Established: **this cached 40-token engine is slower than this 48-token engine**.
- Not established: 40 tokens is inherently slower, all 40-token builds will be slower, or any specific alignment/fusion issue caused it.
- If further investigation is wanted, first rebuild 40 in a NEW cache under current settings and repeat; do not start with custom kernels or architecture changes. This was not performed in the reported check.

Artifacts: `scratch/nn-inference-bench/token-sweep/{results.json,model-comparison.json,run.py}` and trial logs/commands. Neither shape truncates inputs in this benchmark. In real play, states longer than the configured limit must retain full-input fallback.

## Where the approximately 4 ms batch goes

A separate three-trial baseline at 512×48 measured 123.96–129.88k prepacked evaluations/s (median 127.67k), or 3.94–4.13 ms/batch. CPU-fed throughput was 118.50–119.81k. Each mode was timed for 2 seconds per invocation.

One prepared-only Nsight Systems capture covered 497 batches, averaging 4.028 ms/batch:

| Disjoint host wall-time component | ms/batch | share |
|---|---:|---:|
| Wait inside `cudaStreamSynchronize` |3.429|85.1%|
| Input `cudaMemcpy` call |0.218|5.4%|
| Two output `cudaMemcpy` calls |0.135|3.4%|
| `cudaGraphLaunch` call |0.064|1.6%|
| Other host time |0.182|4.5%|

This partitions the host timeline, not independent hardware activities. Transfer APIs include driver/wait overhead; synchronization time cannot be labeled kernel-active time or pure GPU idle time. The residual includes allocations, ORT work, copying, validation and reporting. GPU activity may overlap host calls.

### Device kernel timings

After Windows GPU counter permissions were enabled, Nsight Compute recorded three complete repetitions of a 45-kernel graph. Clocks/cache state were uncontrolled; one timing pass per node. Summed kernel durations: 3.571/3.590/3.587 ms, mean 3.583 ms.

| Approximate kernel family | kernels/graph | mean summed ms/graph | share of summed kernel time |
|---|---:|---:|---:|
| Dense matrix/vector kernels |18|1.762|49.2%|
| Attention kernel |4|0.397|11.1%|
| Normalization/residual fused kernels |13|0.842|23.5%|
| Layout/slicing/embedding/reformat kernels |8|0.577|16.1%|
| Remaining head/mask kernels |2|0.004|0.1%|

Semantic grouping is inferred from kernel symbols, not an exact ONNX-layer mapping; fused kernels span categories. n=3 graphs in one instrumented run, not independent trials. Profiling changes execution, and a separate CPU-only benchmark started around this time. Do not use the profiled application's wall throughput as a clean baseline or subtract this kernel sum from another run to infer exact idle gaps.

A separate single whole-graph capture with seven replay passes reported DRAM throughput 64.42% of peak sustained, SM compute throughput 25.94%, achieved occupancy 44.09%. These are hardware-profiler metrics, not NVML busy or useful model FLOPs/peak FLOPs. Default profiler clock control lowered SM frequency to ~1.31 GHz; its 4.47 ms graph duration is not directly comparable to the unprofiled cycle time. The memory-heavy balance is directional evidence, not proof every kernel is bandwidth-bound.

Artifacts: `scratch/nn-inference-bench/{baseline-1.log,baseline-2.log,baseline-3.log,prepared-profile.nsys-rep,prepared-profile.sqlite,compute-permission-retry.ncu-rep,kernel-times.ncu-rep,kernel-times.csv,kernel-summary.json}`.

## Why the FLOP-only ceiling is not a practical target

For tokens T=48, width d=128, feed-forward width 4d, each transformer layer requires approximately `24*T*d*d + 4*T*T*d` FLOPs, counting multiply-add as 2. Four layers plus input projection/heads total roughly 81 million FLOPs per evaluation, excluding explicitly pricing normalization, softmax, GELU and embeddings.

A historical large-matrix reference on this machine reached about 28 TFLOP/s bf16. Dividing by 81 million gives approximately 346k evaluations/s. That is a **compute-only reference using a different workload**, not an attainable transformer forecast. The original peak-test raw output was not remeasured during this investigation; the 28 TFLOP/s figure comes from the existing project report.

The kernel breakdown explains the limitation of that estimate: a substantial share of runtime is spent on operations poorly represented by a dense-GEMM FLOP budget, including moving intermediate activations. Even the matrix and attention operations differ from one large efficient GEMM. No 200–250k target is established by these measurements.

## Should idle CPU threads also evaluate the network?

A short CPU-only check used the same dynamic FP32 ONNX model, explicit CPUExecutionProvider, one intra/inter-op thread per call, and concurrent callers sharing a session. Inputs had 32 actual tokens (the dynamic CPU path does not add padding to 48). Two 1-second trials per configuration, with prepared synthetic inputs and no concurrent simulator/GPU workload:

| CPU caller threads | Best tested batch per caller | Evaluations/s, observed range |
|---|---:|---:|
| 1 | 8 | 1,741–1,743 |
| 2 | 8 | 3,303–3,350 |
| 4 | 8 | 5,766–5,780 |

Batch sizes 1, 8 and 32 were screened for one caller; 8 and 32 for two/four callers. These ranges are not confidence intervals, and this is not an exhaustive CPU optimization study. Thread affinity was not controlled. Python/ORT call overhead is included; native hybrid queue/simulator contention is not. CPU-only input preparation was outside the timed loop.

Adding those standalone rates to the observed approximately 92k integrated / 131k backend GPU rates gives only about 2.5–6.3% nominal extra throughput for two/four CPU callers. That is arithmetic addition, **not a measured hybrid speedup or a strict upper bound**. Physical cores, caches, memory and dispatch would be shared with simulation, so gains could be smaller or negative. CPU FP32 and GPU FP16 paths can also alter near-tie searches. Idle logical-thread capacity does not imply equally available physical cores.

Recommendation: keep CPU inference for required fallback/testing; do not add hybrid inference complexity for this small potential gain without stronger evidence. Prefer bounded training/strength evaluation and review of replay/export overhead improvements. Raw commands, all tested configurations and results: `scratch/nn-inference-bench/cpu-check/`.

## WSL profiling status

Nsight Compute 2025.1.1 initially returned `ERR_NVGPUCTRPERM`. Enabling GPU performance-counter access in Windows NVIDIA Control Panel resolved it without changing the OS or installing dependencies. Nsight Systems 2024.6.2 still returned host API traces but no usable device-kernel/memory events; counter access did not fix that separate limitation.

No custom kernels or model architecture changes were made. This investigation establishes a reasonable current backend configuration, not an obligation to optimize this exact transformer indefinitely.
