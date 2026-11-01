# Combat NN

Two combat model topologies behind one prediction contract, with Gumbel self-play. Python trains; C++ simulates fights and searches. No dependency on the old agent implementation.

See the [documentation index](docs/README.md), [production inference benchmark](docs/INFERENCE_BENCHMARK.md), and [measured GPU settings and profiling findings](docs/GPU_PERFORMANCE.md).

## Main files

- `play.py`: `play()` runs one `nn_play` process per stage (fresh CUDA context); `Search`/`Workers` settings.
- `learner.py`: `Learner`: network + optimizer; `fit` on replay and `save`/`load` checkpoints (with optimizer and RNG). No deployment policy.
- `play.cpp`: combat `infra::Episode` adapter and CLI. See [WORKERS.md](WORKERS.md).
- `infra/workers`: actors, inference server, supervision, replay readers and prefetch.
- `infra/inference`: CPU/GPU backends; no agent-specific inference implementation.
- `agents/combat/search`: the Gumbel search `play.cpp` drives (`--sims 0` = the network alone) and the rollout blend (`--rollout-mix`).
- `tokens.py`: typed token kinds/card zones and named tensor field access for the native wire format.
- `model.py`: frozen `ModelSpec`, `CombatModel` prediction contract, and the explicit two-topology factory.
- `transformer.py`: `Transformer` attends over all tokens.
- `entity.py`: `EntityNet` pools relics and draw/discard/exhaust cards before attention; scatters predictions back to action positions.
- `export.py`: `ModelExporter`, deployment capacity and precision, and the entity's bounded-attention adapter.
- `graphs.py`: graph/weight reuse, independent of topology and tracing policy.
- `data.py`: binary self-play samples → dense padded tensors.
- `train.py`: outcome BCE + improved-policy cross-entropy, AdamW, bf16 training.

## Model and deployment boundaries

```text
ModelSpec ──► build_model ──► CombatModel
                              ├─ Transformer
                              └─ EntityNet
                                  │
                  ┌───────────────┴────────────────┐
               Learner                        ModelExporter
          optimizer/checkpoint            ExportSpec: capacity/precision
                                                │
                                    dynamic CPU + fixed GPU bank
                                                │
                                      unchanged native search
```

Both models accept `[batch, tokens, 68]` FP32 tokens and return win probabilities
`[batch]` and action logits `[batch, tokens]`. Context is first; non-action positions
are masked below -1e8. Padding never changes real-token predictions. Neither model
owns devices, export precision, worker scheduling or training policy.

Entity prediction is a sequence of small stages: classify groups, embed, pool, compact,
attend, then scatter predictions back to wire positions.

The entity bounded view is an export adapter, not a different checkpoint architecture.
`gpu_attended` counts retained tokens (context, enemy, hand and actions), excluding the
four pooled tokens. Overflow emits a NaN value; the existing native backend retries
against the unbounded FP32 CPU model. It never silently accepts truncated predictions.
This uses the existing nonfinite/precision-retry counter, not a separate overflow metric.

Export works on an independent copy: the training model's device, mode, weights and optimizer
remain untouched, including on failure. Fixed models take FP16 tokens and return FP32
predictions. By default compute precision is left to TensorRT; `gpu_mixed_precision = true`
traces CUDA FP16 autocast explicitly and requires CUDA at export time. CPU/dynamic exports
always remain FP32. Graph reuse is allowed only with an exact mapping for every parameter;
otherwise export retraces (mixed-precision weights often cannot be reused).

Architecture specs explicitly name `transformer` or `entity`. There is no implicit/default
kind, legacy `Net` alias, checkpoint translator or old export-function facade. Old unnamed
transformer checkpoints/configs and historical Python experiment scripts are not supported
by this interface. Entity parameter names and explicit architecture specs are unchanged.
Native weights here are ONNX; the token/action schema and native model filenames are unchanged.

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
- A small job cannot fill these slots: 200 total games still means only 200 games in flight, regardless of `[workers] slots`.
- Default-network GPU starting shape: **batch 512 × 48 tokens × 68 features**. Each batch is padded to this shape. Watch the exit summary's **mean leaves**: low fill wastes GPU work. States exceeding the configured token count use the CPU model, never truncation.
- Fixed-size collections drain: 2,000 starts with 2,000 slots have no replacement reserve. More total games sustain occupancy longer but change games/update. Alternatively, an optional **fixed-shape bank** reduces padded work as the collection drains, without dropping long fights or changing search.
- For the tested **128-wide / 4-layer panache network**, benchmark `gpu_batch = 512`, `gpu_bank = [256, 32]`, `gpu_wait_us = 100` (48 `gpu_tokens`). This exports three same-weight ONNX models; one shared evaluator chooses the smallest engine that fits each gathered batch. Bank engines must have identical weights, token count and feature schema. Extra export/build cost matters on short updates; this is not a universal default. See the [throughput investigation](../../../experiments/25-10-26/a-gumbel/nn/THROUGHPUT_REPORT.md).
- Scheduling uses only infra actors: `[workers] slots` per actor (preset 224), `actors` (preset 9), plus one inference server.
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

Training runs through the app [`apps/gumbel_nn_combat_expert_iteration`](../../../apps/gumbel_nn_combat_expert_iteration/README.md),
whose recipe calls `play()`, `Replay` and `Learner`. Checkpoints hold `net`, `optimizer`, `spec` and the RNG
states; `Learner.load` refuses a different architecture.

The run writes `status.json` live, `log.jsonl` per iteration, and results and `worker.log` per play stage.
Native `gpu_stats` JSON lines report exact real leaves, computed row capacity, engine usage, CPU fallback,
transfer/inference timings. These are host wall-time diagnostics,
not GPU occupancy or percentage of peak FLOPs. With a shape bank, fill is `leaves / capacity`, not mean leaves divided by the largest shape.

Standard dashboard definitions live in [METRICS.md](METRICS.md). Native `play_stats`/`actors_stats` add searched decisions/s and thread CPU counters. `play()` records optional NVML `gpu.csv`; absent hardware counters stay unavailable, never inferred from GPU-busy time. GPU peak bandwidth/compute percentages need profiling, and coordinator CPU/non-CPU percentages are not an I/O-bound classification. The [real-training snapshot](../../../experiments/25-10-26/a-gumbel/nn/LOCAL_TRAIN_REPORT.md) includes phase times, observed device activity and current packing bottleneck. See the panache experiment dashboard under `experiments/25-10-26/a-gumbel/nn/`.

### Binary samples

Each searched decision: int32 token count, int32 action count, float32 game outcome, `tokens × 68` float32 features, then one float32 policy probability per action. Action tokens are last. Replay uses FP32 minibatches to preserve numeric range and exact token IDs. Forced moves are not training rows. Cap outcomes (50 turns / 512 actions) count as non-wins.

## Scope

This is a simple Ironclad combat pipeline, not a learned simulator. It reuses the environment encoder, native simulator and public-belief particle sampler. Guided rollouts remain mixed into search values (default 0.5) because they helped the tested agents. Search settings, batch shape and model topology remain explicit. Unsupported states fail rather than silently dropping information.

Replay uses native result counts to index the unchanged binary samples by deck. It selects at most
`max_train_rows` compact row addresses, samples uniformly (or evenly across decks with `balanced_replay`),
and pads/transfers only length-bounded FP32 microbatches. Their losses are weighted by
row count and gradients accumulated before one effective-batch optimizer step; no rows
are truncated or dropped. Per-update `replay.json` records available,
selected and actually drawn rows per deck. Index caches expire with their samples.
Native results now report explicit episode caps and completion timestamps. `nn_play check`
validates/encodes starts without network inference, search, or game outcomes.
Nonfinite GPU predictions are retried once against the original FP32 CPU model; invalid
FP32 results still fail. Long CPU fallback batches also have bounded attention shapes.

## Cutover

The command-line loop (`loop.py`) is gone: the app owns the recipe and its typed config. Runs started
by it cannot be resumed; start a new run with `[run].initial_checkpoint` to reuse weights, optimizer and RNG.
Native `nn_play import` remains a standalone sample conversion tool.

Historical performance reports describe the removed implementation; current
component benchmarks live in `infra/benchmarks` (`make infra-benchmark`).
