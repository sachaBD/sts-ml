# Standard runtime metrics

Machine-readable definitions are in `metrics.py::METRIC_DEFINITIONS`. Missing counters are `None` / dashboard `—`, never zero or inferred from a different kind of utilization.

| Metric | Denominator / interpretation |
|---|---|
| Decisions/s | **Searched game decisions** / gameplay wall seconds. Not forced actions or NN leaves. |
| NN evaluations/s | Real evaluated leaves (GPU + CPU fallback) / gameplay wall seconds. Padding excluded. |
| Games/h, gameplay | Finished games / gameplay seconds ×3,600. |
| Games/h, full update | Finished games / collection + replay loading + training + checkpoint/export wall time ×3,600. Includes per-update engine setup. Initial initialization/export is reported separately. |
| Worker CPU % | Sum SimWorker **thread CPU time** / (allocated worker count ×whole gameplay wall time). Includes idle tail capacity in the denominator. |
| Coordinator CPU % | Coordinator thread CPU time / gameplay wall time. Includes CPU consumed polling inside driver calls; not just useful CPU computation. |
| Coordinator GPU API % | Host wall time inside H2D + inference + D2H calls / gameplay wall time. Overlaps CPU time; the percentages need not add to100. |
| Coordinator non-CPU % |100−CPU%. Includes descheduling/waits; **not an I/O-bound percentage**. Pure I/O-bound fraction requires tracing and is unavailable. |
| GPU busy % | NVML percent **time** kernels were active. Not fraction of peak SM/Tensor FLOPs. |
| GPU VRAM % | Device-wide used/reserved memory / reported capacity. Includes desktop, runtime contexts and cached training allocations. Not bandwidth efficiency. |
| GPU memory activity % | NVML percent time global memory was read/written. Retained in JSON, not called bandwidth utilization. |
| GPU memory bandwidth % | Requires profiling counters and a defined peak-bandwidth reference; unavailable in low-overhead routine reporting. |
| GPU compute throughput % | Requires profiling counters and a defined SM/compute peak. Unavailable in low-overhead routine reporting; NVML GPU busy is not substituted. |

`gpu_stats.busy` is evaluator host wall time (transfers, inference API calls, copying/results); it must NOT be displayed as hardware GPU busy. Likewise, NN evaluations/s is not decisions/s.

## Capture and dashboard

- Native `play_stats` reports searched decisions, completed games, gameplay seconds and start epoch.
- Native `runtime_stats` reports worker/coordinator thread CPU counters (where `CLOCK_THREAD_CPUTIME_ID` is supported). Unsupported platforms omit CPU fields.
- `Run.play` captures `iterNNN/gpu.csv` with optional nvidia-smi sampling. Absent nvidia-smi leaves counters unavailable. The short training benchmark instead records one continuous sampler across all phases, avoiding duplicate samplers.
- NVML is polled every250ms; its utilization windows overlap and are not independent observations. Means are approximate phase/time-activity estimates. Do not turn sample counts into statistical confidence intervals.
- `dashboard.py RUN_DIR` reads the same metric definitions and stored counters. It supports both normal loop runs and the short training benchmark. Historical runs without these counters show missing values.
- Device means use native gameplay timestamps, excluding engine setup. Separate benchmark phase summaries cover training, replay loading and export. Host clock disruptions invalidate timestamp alignment; preserve raw data and flag affected runs.

A high GPU-busy percentage can coexist with low peak-compute throughput. A coordinator can consume a full CPU core while spending much of its wall time inside GPU API calls. Neither justifies claiming hardware compute efficiency or I/O-bound time without profiling.
