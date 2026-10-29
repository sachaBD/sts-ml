# Combat NN

A standalone transformer + Gumbel self-play pipeline. Python trains; C++ simulates fights and searches. No dependency on the old agent implementation.

See the [documentation index](docs/README.md), [production inference benchmark](docs/INFERENCE_BENCHMARK.md), and [measured GPU settings and profiling findings](docs/GPU_PERFORMANCE.md).

## Main files

- `loop.py`: play → train on disk-indexed replay → export. Launches **one** C++ process.
- `play.cpp`: combat `infra::Episode` adapter and CLI. See [WORKERS.md](WORKERS.md).
- `infra/workers`: actors, inference server, supervision, replay readers and prefetch.
- `infra/inference`: CPU/GPU backends; no agent-specific inference implementation.
- `gumbel.cpp`: search pauses at each unevaluated leaf. Root sequential halving with Gumbel noise; deterministic improved-policy selection inside the tree. Each simulation samples a public-belief particle.
- `model.py`: 2-layer, width-64, 2-head transformer. Context token produces win probability; legal-action tokens produce policy logits.
- `data.py`: binary self-play samples → dense padded tensors.
- `train.py`: outcome BCE + improved-policy cross-entropy, AdamW, bf16 training.

## Getting good hardware utilisation

### Training

- Disk-indexed replay stays disk-indexed;
  transfer only FP32 minibatches, with BF16 autocast and length-bounded gradient accumulation.
- Use **bf16 mixed precision** and a substantial batch (default **2,048**). Tiny batches spend their time launching kernels, not doing useful work.
- Keep diagnostics off the hot path: copy loss statistics to the host only when logging.
- Budget training by **steps**, not repeated full passes over an expanding replay. Current recipe is 100 steps × 2,048 states after 1,000 games, training on the last five iterations.
- Dense padding is intentionally simple but costs memory; longer states reduce throughput.

### Self-play / inference

- Use **one GPU evaluator**, not one GPU session per CPU thread or process. Small independent GPU calls are slower than CPU inference.
- Keep enough games in flight. Starting point: **9 actors × 224 slots = 2,016 games**.
- A small job cannot fill these slots: 200 total games still means only 200 games in flight, regardless of `--parallel`.
- Default-network GPU starting shape: **batch 512 × 48 tokens × 68 features**. Each batch is padded to this shape. Watch the exit summary's **mean leaves**: low fill wastes GPU work. States exceeding the configured token count use the CPU model, never truncation.
- Fixed-size collections drain: 2,000 starts with 2,000 slots have no replacement reserve. More total games sustain occupancy longer but change games/update. Alternatively, an optional **fixed-shape bank** reduces padded work as the collection drains, without dropping long fights or changing search.
- For the tested **128-wide / 4-layer panache network**, benchmark `--gpu-batch 512 --gpu-bank 256,32 --gpu-wait-us 100` (48 tokens). This exports three same-weight ONNX models; one shared evaluator chooses the smallest engine that fits each gathered batch. Bank engines must have identical weights, token count and feature schema. Extra export/build cost matters on short updates; this is not a universal default. See the [throughput investigation](../../../experiments/25-10-26/a-gumbel/nn/THROUGHPUT_REPORT.md).
- Scheduling uses only infra actors: `--parallel` slots/actor (default 224), `--workers` actors (default 9), plus one inference server.
- Export `model-gpu.onnx` with **fixed shapes and fp16 input**. TensorRT uses fp16 internally; outputs stay fp32. Dynamic shape computations can fall back to CPU and prevent graph capture.
- **TensorRT fusion** reduces kernel count and memory traffic. **CUDA graphs** reduce CPU launch overhead. They solve different problems; both are enabled.
- Padding and fp16 conversion run on CPU workers. Reusable buffers avoid allocation every request; the GPU thread concatenates prepared chunks into pinned memory and transfers half as many input bytes as fp32.
- Build engines before gameplay starts, with no training running on the GPU. Engine cache lives beside each model (`tensorrt/`). **Kernel timing cache is shared** at `$XDG_CACHE_HOME/stsrl/tensorrt-timing` (default `~/.cache/stsrl/tensorrt-timing`), with builder optimization level **3**. This avoids repeated tactic measurement for each model. Timing caching reduces variability; it does not guarantee deterministic tactics. Level 5 was much slower to build and did not fix variability.
- CPU usage need not be 100%: once simulation can feed the GPU, waiting for inference is expected. Judge **completed games per wall-clock second**, not utilisation percentages alone.

### Measurements on this RTX 3060 / WSL machine

Historical measurements use the same panache model at 256 simulations. Game lengths, token counts and batch fill affect results.

| Configuration | Observed throughput |
|---|---:|
| CPU inference, cross-game batching | ~21,100 games/hour (1,000 games) |
| TensorRT fp16 + graphs, fp32 transport, 2,000 games in flight | ~50,500 games/hour (2,000 games) |
| New fp16 transport + shared timing cache, network-only | ~374–377k evaluations/second (two fresh cache-backed builds; three timed blocks each) |

**Do not equate evaluations/s with games/s.** The newest worker-packing/fp16-transport path has passed a short GPU smoke check; full-game throughput has not been remeasured after these changes. 80–100k games/hour was an estimate, not an established result.

Large standalone GPU tests reached ~28 TFLOP/s bf16 and ~328 GB/s device-memory bandwidth. The hardware is healthy; this small network cannot approach those peaks because its operations are small and memory-heavy.

## Build

Dependencies are in `pyproject.toml` / `uv.lock`: PyTorch, ONNX, ONNX Runtime GPU 1.24.4 and TensorRT CUDA-12 libraries. Use the managed environment, not direct pip installs:

```sh
make env-all  # or env-onnx if JAX isn't needed
```

Install a **CUDA 12 toolkit** for C++ headers and `libcudart` (this machine uses `/usr/local/cuda`). Later ONNX Runtime releases tested here expected CUDA 13 and did not match the CUDA 12 environment.

Prepare ONNX Runtime C++ headers matching the installed wheel (generated files only):

```sh
version=$(.venv/bin/python -c 'import onnxruntime; print(onnxruntime.__version__)')
mkdir -p build/deps/onnxruntime/include
for h in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h onnxruntime_ep_c_api.h; do
    curl -fLsS "https://raw.githubusercontent.com/microsoft/onnxruntime/v${version}/include/onnxruntime/core/session/$h" \
        -o "build/deps/onnxruntime/include/$h"
done
cmake -S . -B build/nn -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=/usr/local/cuda
cmake --build build/nn --target nn_play --parallel 6
```

## Run

For managed multi-deck Champ training, use
[`apps/gumbel_nn_combat_expert_iteration`](../../../apps/gumbel_nn_combat_expert_iteration/README.md).
It loads ten-deck starts and historical monitoring scores, copies inputs/the worker,
and delegates training to this loop. `--start` also accepts a list of named starts;
`--games` then means the total, divided equally and interleaved across those starts.
`--play` can select a frozen native executable (default remains `build/nn/.../nn_play`).

`loop.py` exports both CPU and GPU models and configures the installed TensorRT/CUDA library paths automatically.

```sh
.venv/bin/python -m agents.combat.nn.loop \
    --start START.json --out runs/YOUR_NEW_RUN \
    --iterations 10 --games 2000 --steps 100 --batch 2048 --workers 9 \
    --gpu-batch 512 --gpu-tokens 48 --sims 256
```

Topology can be selected with `--dim`, `--layers`, and `--heads` (defaults: 64/2/2).
New checkpoints include a `spec` dictionary; reconstruct with `Net(**checkpoint['spec'])` before loading weights.
Legacy checkpoints without `spec` retain the default topology. `model-spec.json` also records the run's architecture.

This is an example, not a new approved experiment. `START.json` contains `{fight_id, start}`. Output directories must be new unless explicitly continuing with `--resume-from N`.
Resume requires the latest completed iteration's checkpoint and existing replay. `--games`, `--iterations`,
`--test-every`, `--prune-replay` and `--max-train-rows` may change. Archive any interrupted next-iteration directory first; partial samples are never appended.
Stop the old process before resuming. New checkpoints restore training RNG state; legacy checkpoints without
RNG state restart sampling from seed 0 (not a bit-exact continuation). Keep checkpoints, samples, engines and generated inputs under `runs/` or `scratch/`.

For unattended runs, `--iterations 0` runs until `OUT/STOP` exists, finishing the active update and
scheduled evaluation before exiting. `--test-every N` adds recurring evaluations. `--prune-replay`
removes samples outside the replay window and old TensorRT caches, retaining models and results.
`--max-train-rows N` caps uniformly selected indexed rows; 0 indexes all rows. Only minibatches are read.
A process-held `.lock` prevents overlapping loops in one output directory.
`--export-reuse` runs the ONNX exporter once per shape and then rewrites weights.

The run writes `status.json` live, `log.jsonl` per iteration, and results and `worker.log` per play stage.
Native `gpu_stats` JSON lines report exact real leaves, computed row capacity, engine usage, CPU fallback,
transfer/inference timings. These are host wall-time diagnostics,
not GPU occupancy or percentage of peak FLOPs. With a shape bank, fill is `leaves / capacity`, not mean leaves divided by the largest shape.

Standard dashboard definitions live in [METRICS.md](METRICS.md). Native `play_stats`/`actors_stats` add searched decisions/s and thread CPU counters. `Run.play` records optional NVML `gpu.csv`; absent hardware counters stay unavailable, never inferred from GPU-busy time. GPU peak bandwidth/compute percentages need profiling, and coordinator CPU/non-CPU percentages are not an I/O-bound classification. The [real-training snapshot](../../../experiments/25-10-26/a-gumbel/nn/LOCAL_TRAIN_REPORT.md) includes phase times, observed device activity and current packing bottleneck. See the panache experiment dashboard under `experiments/25-10-26/a-gumbel/nn/`.

### Binary samples

Each searched decision: int32 token count, int32 action count, float32 game outcome, `tokens × 68` float32 features, then one float32 policy probability per action. Action tokens are last. Replay uses FP32 minibatches to preserve numeric range and exact token IDs. Forced moves are not training rows. Cap outcomes (50 turns / 512 actions) count as non-wins.

## Scope

This is a simple Ironclad combat pipeline, not a learned simulator. It reuses the environment encoder, native simulator and public-belief particle sampler. Guided rollouts remain mixed into search values (default 0.5) because they helped the tested agents. Search settings, batch shape and model topology remain explicit. Unsupported states fail rather than silently dropping information.

`--balanced-replay` selects equal deck quotas instead of uniform row sampling.
Replay uses native result counts to index the
unchanged binary samples by deck. It selects at most `--max-train-rows` compact
row addresses, samples uniformly (or evenly across decks with `--balanced-replay`),
and pads/transfers only length-bounded FP32 microbatches. Their losses are weighted by
row count and gradients accumulated before one effective-batch optimizer step; no rows
are truncated or dropped. Per-update `replay.json` records available,
selected and actually drawn rows per deck. Index caches expire with their samples.
`--initial-checkpoint` starts a new experiment from matching weights, optimizer and
RNG without inheriting the old replay. `--validation-share 0.20` schedules validation
at update boundaries when its cumulative recorded wall share falls below20%; the
initial test is included. Explicit `--test-at` evaluations override this budget.
`--test-in` adds separate in-sample validation under each test stage; both phases count
against the wall budget, only `--test` is the primary score. `--min-free-gib` stops cleanly
before collection when the reserve plus one estimated batch will not fit.
Native results now report explicit episode caps and completion timestamps. `nn_play check`
validates/encodes starts without network inference, search, or game outcomes.
Nonfinite GPU predictions are retried once against the original FP32 CPU model; invalid
FP32 results still fail. Long CPU fallback batches also have bounded attention shapes.

## Infrastructure cutover

There is no legacy scheduler or dense/cached replay path in the loop. Old
`--scheduler`, `--groups`, `--replay-cache` and bootstrap-loop
options are rejected. Start a new run with `--initial-checkpoint` to reuse weights,
optimizer and RNG; do not resume an old command with the new binary. Native
`nn_play import` remains a standalone sample conversion tool, not a loop input.

Historical performance reports describe the removed implementation; current
component benchmarks live in `infra/benchmarks` (`make infra-benchmark`).
