# Combat NN

A standalone transformer + Gumbel self-play pipeline. Python trains; C++ simulates fights and searches. No dependency on the old agent implementation.

## Main files

- `loop.py`: bootstrap from stored MCTS games, then play → train → export. Launches **one** C++ process.
- `play.cpp`: `main()` starts CPU threads; `worker()` alternates two groups of games. While one group waits for inference, the thread advances the other.
- `gpu.cpp`: one GPU thread collects leaves from every CPU worker and evaluates a fixed-size TensorRT batch. `submit()` pads/converts inputs on the calling CPU worker; `loop()` batches; `run()` transfers and evaluates.
- `gumbel.cpp`: search pauses at each unevaluated leaf. Root sequential halving with Gumbel noise; deterministic improved-policy selection inside the tree. Each simulation samples a public-belief particle.
- `model.py`: 2-layer, width-64, 2-head transformer. Context token produces win probability; legal-action tokens produce policy logits.
- `data.py`: binary self-play samples → dense padded tensors.
- `train.py`: outcome BCE + improved-policy cross-entropy, AdamW, bf16 training.

## Getting good hardware utilisation

### Training

- Keep the replay on the GPU during each training phase; do not transfer every minibatch.
- Use **bf16 mixed precision** and a substantial batch (default **2,048**). Tiny batches spend their time launching kernels, not doing useful work.
- Keep diagnostics off the hot path: copy loss statistics to the host only when logging.
- Budget training by **steps**, not repeated full passes over an expanding replay. Current recipe is 100 steps × 2,048 states after 1,000 games, training on the last five iterations.
- Dense padding is intentionally simple but costs memory; longer states reduce throughput.

### Self-play / inference

- Use **one GPU evaluator**, not one GPU session per CPU thread or process. Small independent GPU calls are slower than CPU inference.
- Keep enough games in flight. Starting point: **10 CPU threads × 200 games/thread = 2,000 games**, split into two groups per thread.
- A small job cannot fill these slots: 200 total games still means only 200 games in flight, regardless of `--parallel`.
- GPU starting shape: **batch 1,024 × 48 tokens × 68 features**. Each batch is padded to this shape. Watch the exit summary's **mean leaves**: low fill wastes GPU work. States exceeding 48 tokens use the CPU model, never truncation.
- Export `model-gpu.onnx` with **fixed shapes and fp16 input**. TensorRT uses fp16 internally; outputs stay fp32. Dynamic shape computations can fall back to CPU and prevent graph capture.
- **TensorRT fusion** reduces kernel count and memory traffic. **CUDA graphs** reduce CPU launch overhead. They solve different problems; both are enabled.
- Padding and fp16 conversion run on CPU workers. Reusable buffers avoid allocation every request; the GPU thread concatenates prepared chunks into pinned memory and transfers half as many input bytes as fp32.
- Build engines before gameplay starts, with no training running on the GPU. Engine cache lives beside each model (`tensorrt/`). **Kernel timing cache is shared** at `build/nn/tensorrt-timing/`, with builder optimization level **3**. This avoids repeated tactic measurement for each model. Timing caching reduces variability; it does not guarantee deterministic tactics. Level 5 was much slower to build and did not fix variability.
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

`loop.py` exports both CPU and GPU models and configures the installed TensorRT/CUDA library paths automatically.

```sh
.venv/bin/python -m agents.combat.nn.loop \
    --start START.json --bootstrap MCTS_GAMES.jsonl --out runs/YOUR_NEW_RUN \
    --iterations 10 --games 2000 --steps 100 --batch 2048 --workers 10 \
    --gpu-batch 1024 --gpu-tokens 48 --sims 256
```

This is an example, not a new approved experiment. `START.json` contains `{fight_id, start}`. Bootstrap JSON lines contain recorded `{fight, search}` results. Output directories must be new; no automatic resume/overwrite. Keep checkpoints, samples, engines and generated inputs under `runs/` or `scratch/`.

The run writes `status.json` live, `log.jsonl` per iteration, and results and `worker.log` per play stage. See the panache experiment dashboard under `experiments/25-10-26/a-gumbel/nn/`.

### Binary samples

Each searched decision: int32 token count, int32 action count, float32 game outcome, `tokens × 68` float32 features, then one float32 policy probability per action. Action tokens are last. The loader pads densely and stores tokens as fp16. Forced moves are not training rows. Cap outcomes (50 turns / 512 actions) count as non-wins.

## Scope

This is a simple Ironclad combat pipeline, not a learned simulator. It reuses the environment encoder, native simulator and public-belief particle sampler. Guided rollouts remain mixed into search values (default 0.5) because they helped the tested agents. Search settings, batch shape and model topology remain explicit. Unsupported states fail rather than silently dropping information.
