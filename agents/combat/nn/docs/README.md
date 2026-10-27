# Combat NN documentation

Durable component documentation and reviewed performance findings. Measurements apply to the specified model, hardware and engine builds, not every architecture.

- [Assessment and next steps](ASSESSMENT_AND_NEXT_STEPS.md): decisions from the throughput review, deferred work, and checks before training.
- [Inference benchmark](INFERENCE_BENCHMARK.md): production-backend benchmark, bounded runtime, modes and profiling.
- [GPU performance findings](GPU_PERFORMANCE.md): batch/token sweeps, kernel breakdown, interpretation and limitations.
- [Worker ownership and scheduling](../WORKERS.md): contracts, cancellation and verification.
- [Runtime metric definitions](../METRICS.md): denominators and utilization semantics.
- [CPU-overhead handoff](../OVERHEAD_SPEC.md): separate replay/export optimization workstream; implementation/results require independent review.

Entry point: [combat NN README](../readme.md). Raw logs, profiler traces, generated models and caches remain outside this documentation directory, under `scratch/` or `runs/`. Local artifact paths in reports are repository-root relative and may not exist in a fresh checkout.
