# Local-worker real-training benchmark — 27 Oct 2026

## Outcome

The new worker-local scheduler completed real self-play → train → checkpoint/export updates successfully. **Meaningful throughput is roughly64k NN evaluations/s and90k–101k games/h including training/export**, depending on whether initial engines are fresh or cached. This is a short operational snapshot, not a play-strength result or a controlled comparison with the old scheduler.

The main actionable bottleneck is **serial input packing on the coordinator**, not insufficient GPU batch fill. Packing consumed15.7s of47.6s gameplay with91% row fill. More active games alone does not address that.

## Actual training, not a frozen-network microbenchmark

Configuration:128 width /4 layers /4 heads; Gumbel64; rollout mix0.5;9 SimWorkers +1 coordinator;224 slots/worker (at most2,016 active games);2,000 games/update; GPU batches512/256/32;48 tokens;100us batch wait. Each update uses100 AdamW training steps ×2,048 states, bf16 mixed precision, lr0.001, and a rolling five-iteration replay (~255k states). Starts have fresh deterministic seeds; every game completes under existing combat caps.

Both benchmarks warm-start the prior iteration10 checkpoint, optimizer and replay. They are separate forks, not three consecutive updates of one model:

| Run | Completed work | Wall time | Training evidence |
|---|---|---:|---|
| Main,2 updates |4,000 games;200 steps |168.56s including initialization/export |optimizer3000→3200; max parameter change0.09566 |
| Instrumented follow-up,1 update |2,000 games;100 steps |72.46s including checkpoint load |optimizer3000→3100; max parameter change0.05798 |

The follow-up reuses validated matching initial model/engine files; its first engine setup is cached. Its higher full-update games/h is NOT evidence of a scheduler improvement over the main run. The main run exports initial models and builds fresh engines. Therefore do not pool their end-to-end times as identical trials. Source checkpoints/replay are untouched; new weights/samples are saved separately. No held-out strength test was run.

## Standard metrics: instrumented follow-up

n=1 full update for thread-level split;187 correlated NVML polls during gameplay. Definitions are shared with the dashboard in `agents/combat/nn/METRICS.md` and `metrics.py`.

| Metric | Observed |
|---|---:|
| Searched decisions/s, gameplay |1,112 |
| Real NN evaluations/s, gameplay |64,653 |
| Games/h, gameplay only |151,275 |
| Games/h, full update |100,698 |
| SimWorker CPU utilization |26.9% of allocated9-thread capacity (~2.42 logical cores) |
| Coordinator CPU utilization |98.7% of one logical core |
| Coordinator wall time in GPU APIs |56.9% |
| Coordinator non-CPU time |1.3%; **not an I/O-bound fraction** |
| Pure coordinator I/O-bound fraction |Not measured; requires tracing |
| GPU busy time |50.4% |
| GPU VRAM occupancy, gameplay mean |8.8% of12,287MiB reported capacity |
| GPU memory activity |50.3% time; **not peak-bandwidth utilization** |
| GPU peak memory-bandwidth / compute throughput |Not measured; profiler counters required |
| Real GPU row fill |91.2% |
| Time to finish last10% of games |2.60s |

Coordinator CPU and GPU-API percentages overlap: driver calls can consume CPU while waiting/polling. Thus98.7% CPU does not prove98.7% useful CPU work, and1.3% non-CPU does not mean only1.3% I/O-bound time. GPU busy is time, not fraction of the card's ideal math throughput. Peak-bandwidth/compute fields deliberately remain unavailable rather than fabricated.

## Main run variability and phase costs

Main run n=2 sequential updates, different trained weights and fresh starts. Ranges are observed variation, not confidence intervals:

- Gameplay47.43–47.90s for2,000 games;63.8–64.1k NN evaluations/s;1,100–1,101 searched decisions/s.
- Full updates78.03–80.15s:89.8–92.3k games/h. Aggregate91.0k/h excluding initial initialization/export;85.4k/h including the full168.56s workflow.
- Fresh per-update engine setup8.09–8.92s; replay loading4.29–5.04s; actual training8.41–8.97s; save/export8.03–8.17s.
- Training GPU busy90.6–95.1% (35/33 overlapping250ms polls). This is active time, not near-peak Tensor utilization.
- Mean board power88.7–89.5W during collection including setup,114.9–116.5W during training (n=2 phases each). Power is not compute-throughput efficiency.
- Whole-device peak VRAM observed5,139MiB (~42% capacity), including cached training allocations. VRAM capacity is not the current bottleneck.
- Main run last10% tail2.88–2.90s. No host clock disruption detected (wall/monotonic drift under0.01s).

Follow-up full update71.50s splits into49.22s collection including cached startup,4.36s replay load,8.86s train,9.05s save/export (small cleanup/accounting differences). The8.86s train phase averages91.3% GPU busy across35 overlapping polls.

## Where gameplay time goes

Follow-up GPU evaluator host timing, not hardware occupancy:

- Input packing/conversion/padding:15.72s (~33% of gameplay), outside evaluator busy timer.
- Inference API calls:23.80s (~50%).
- H2D/D2H calls:3.29s (~7%).
- Concatenation:1.52s; result scatter/fulfillment:0.67s.

These stages execute serially on the coordinator in v1. The metrics do not separate conversion, allocation and padding within packing. They do establish that the CPU→NN preparation path is a substantial serialization point. The state/search ownership design works; it is not evidence that moving full simulator states between cores is the bound (we do not move them).

**Recommendation:** target packing next—vectorized FP16 conversion and/or worker-owned prepared input buffers—before increasing slots/workers or optimizing GPU-fill percentages further. Preserve immutable-input lifetimes and benchmark end-to-end. Export/build and replay-load overhead are also material; a gameplay-only speedup is not automatically a training-loop win. No speedup from a proposed change is claimed yet.

## Reproduce / inspect

```
PYTHONPATH=. .venv/bin/python experiments/25-10-26/a-gumbel/nn/train_throughput.py \
  --out runs/YOUR_NEW_BENCH --iterations 2 --games 2000 --steps 100

watch -n 5 '.venv/bin/python experiments/25-10-26/a-gumbel/nn/dashboard.py runs/nn-local-train-bench-metrics-20261027'
```

Generated artifacts:
- `runs/nn-local-train-bench-20261027/`: main report.json, phases.jsonl, GPU CSV, native logs/results/samples and trained checkpoints.
- `runs/nn-local-train-bench-metrics-20261027/`: follow-up report.json, derived standard-report.json, complete thread/device counters and new checkpoint.
- Corresponding sibling `.log` files capture console output.

The benchmark uses production Run.play/train/export functions. The small orchestration harness avoids spending the benchmark budget on fresh bootstrap training. Standard counter tests, native ownership/failure tests and sampler smoke pass. Raw reports and source metrics distinguish missing counters from zero utilization. Profiler-based card-peak percentages remain unverified.
