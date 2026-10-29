> Historical measurements of the pre-infra implementation. Current backends and
> component benchmarks are in `infra/`; the old native benchmark was removed.

# Production GPU throughput benchmark

`nn_bench_inference` uses `gpu_evaluator` directly: production ONNX Runtime/TensorRT FP16 fusion, optimization, cached engines and CUDA graphs, synchronous coordinator-owned evaluation. It includes production pinned-buffer concatenation, transfers, result allocation/scatter and prediction validation. No simulator/search/training. The experimental staged-pipeline implementation was assessed separately and removed; this benchmark retains stable synchronous evaluation. This measures the backend, not a pure GPU-kernel ceiling.

Build: `cmake --build build/nn --target nn_bench_inference --parallel 2` (reconfigure CMake first for a newly added target).

Run with existing matching CPU and fixed-shape GPU ONNX models. Architecture comes from the supplied model, not benchmark constants. Example from the repository root:

```sh
packages="$PWD/.venv/lib/python3.13/site-packages"
export LD_LIBRARY_PATH="$packages/tensorrt_libs:$(find "$packages/nvidia" -type d -name lib | paste -sd:):${LD_LIBRARY_PATH:-}"
timeout -k 1s 59s build/nn/agents/combat/nn/nn_bench_inference \
  runs/nn-local-train-bench-20261027/iter000/model.onnx \
  runs/nn-local-train-bench-20261027/iter000/model-gpu.onnx \
  --batch 512 --tokens 48 --state-tokens 32 --seconds 3
```

Default hard process deadline is **55 seconds**, including engine setup, three warm-up calls, both measurements and teardown. `--limit-seconds` may not exceed 60; the external timeout adds protection in case of a watchdog failure. Cold engine construction may hit the deadline: this is a failure (exit 124), not a throughput result. Prefer already cached engines; never start unbounded engine builds for this benchmark.

For Nsight capture, add `--profile-prepared 1` and use `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop ...`. This brackets only the warmed-up prepared-input measurement using CUDA profiler start/stop calls. Device profiling availability depends on the host/driver/permissions; host API tracing alone cannot distinguish active GPU kernels from GPU idle gaps.

Modes (always both, separately timed):
- `prepared`: repeatedly evaluate one prepacked batch. Backend reference; a fresh vector of input views and output predictions is still created each call, as in production.
- `cpu_fed`: one producer thread generates fresh synthetic tokens and calls production FP16 preparation. The consumer evaluates full batches using a bounded pool (`--queue`, default 3 total buffers, including buffers held by either thread). Includes initial queue fill; no unlimited precomputation.

Tokens have valid type/embedding-index ranges, context first, up to four action tokens last, bounded numeric values and zero padding to the GPU shape. They are NOT semantically valid combat states or a representative distribution. Synthetic random-number generation is included in CPU-fed cost and can itself be a bottleneck. `--state-tokens` is fixed per invocation. No truncation, partial batches, shape bank or CPU fallback. GPU model signature must match `--batch` and `--tokens`.

JSON stdout rows contain measured evaluations/s, batches, wall seconds, mean batch cycle time (including producer wait in CPU-fed mode), full row fill, shapes and producer/consumer queue wait seconds. Setup timing is a separate text line. Producer wait is read after join and includes a small shutdown boundary outside the timed consumer interval. Native GPU timing reports go to stderr and include warm-up plus both modes; do not interpret them as per-mode counters. Output validation checks cardinality, finite legal logits, and finite values in [0,1]. No GPU-utilization percentage is used as a throughput proxy.

Measurements are single short trials, not confidence bounds. Compare repeated invocations if estimating variability; retain logs and keep the same model/cache/shape/settings. Neither sample quality nor playing strength is measured. No automatic benchmark is added to CTest.
