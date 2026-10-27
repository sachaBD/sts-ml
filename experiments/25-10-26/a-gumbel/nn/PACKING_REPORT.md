# Worker-side packing v2 — 27 Oct 2026

Move FP16 conversion/padding to reusable SimWorker slot buffers, immutable until prediction completion. Coordinator consumes views and copies directly to pinned batch memory, without another intermediate packed copy. F16C SIMD conversion where supported; canonical CUDA-host scalar fallback otherwise. CPU fallback for oversized states remains full-input, never truncation. No checkpoint/token/sample-format change.

## Matched real-training comparison

One complete update/version: identical initial model/optimizer/replay, byte-identical2,000 fight starts,64 simulations,9 SimWorkers ×224 slots, GPU512/256/32 ×48 tokens,100us wait,100 training steps ×2,048 states. Both initial model caches warm. New trained model saved; optimizer3000→3100.

| Metric | v1 | v2 |
|---|---:|---:|
| Searched decisions/s |1,112|1,577|
| Real NN evaluations/s |64,653|91,706|
| Games/h, gameplay |151,275|214,566|
| Games/h, full update |100,698|119,981|
| Full update seconds |71.50|60.01|
| Worker CPU, allocated capacity |26.9%|40.7%|
| Coordinator CPU |98.7%|98.3%|
| Coordinator GPU API wall time |56.9%|81.1%|
| GPU busy time |50.4%|71.6%|
| GPU VRAM occupancy, gameplay |8.8%|8.8%|
| GPU row fill |91.2%|91.2%|
| Coordinator packing seconds |15.72|0|

NN throughput improves41.8%; full-update games/h19.1%. Gameplay47.60→33.56s. Entire v2 workflow61.48s including initial checkpoint load. Worker packing totals2.91s summed across workers (overlapped, not additive to gameplay wall).

n=1 matched run/version, not confidence intervals. This combines SIMD and worker-side preparation; no attempt to isolate their individual gains. FP16 batching can change near-tie search paths, so matching starts/packed values does not promise identical trajectories. Actual NN counts differ by85 out of~3.08M (<0.003%). No strength claim. GPU busy is active time, not peak compute; peak compute/bandwidth and pure I/O-bound percentages remain unmeasured. Coordinator CPU includes driver polling.

## Checks and artifacts

- Packing matches CUDA reference bit-for-bit for finite half values, neighboring float values and~100k random floats; output bounds checked.
- Existing worker tests plus prepared-input ownership/lifetime test pass.
- Actual2,000-game collection and100-step training update completed successfully.
- v1: `runs/nn-local-train-bench-metrics-20261027/`.
- v2: `runs/nn-worker-packing-bench-20261027/` (report.json, native logs, samples, updated weights).
- Tests/build: `scratch/nn-worker-packing/`.

Reproduction uses `train_throughput.py --iterations 1 --games 2000 --steps 100 --initial-model runs/nn-local-train-bench-20261027/iter000/model.onnx --fight-namespace nn-local-train-bench-metrics-20261027 --out runs/NEW_DIRECTORY`.
