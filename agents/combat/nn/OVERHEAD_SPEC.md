# CPU-only workstream: reduce per-update orchestration overhead

## Objective and authority

Reduce repeated CPU-side work in the combat NN play/train/export loop without changing learning semantics or touching GPU inference execution. This document is an implementation handoff, not permission to launch training, use the GPU, or modify existing runs.

Success means less measured replay-load and export time across successive updates, with equivalent training inputs and exported model predictions. Do not promise a total pipeline speedup before measuring it.

## Evidence / baseline

Read these first:
- `loop.py`: `Run.play`, nested `fit`, resume validation.
- `data.py`: `read` reparses every file and repads the whole rolling replay.
- `model.py`: `export` and `export_static` repeatedly invoke the PyTorch ONNX exporter and move the training model to CPU.
- `train.py`: uniform sampling with replacement; input upload occurs inside `train`.
- `test_loop.py`, `test_model.py`, `WORKERS.md`.
- `experiments/25-10-26/a-gumbel/nn/train_throughput.py` and `PACKING_REPORT.md`.

One worker-packing benchmark (`runs/nn-worker-packing-bench-20261027/report.json`) measured approximately 36.7 s collection including setup, 5.2 s replay load, 9.1 s training, and 9.1 s save/export. n=1; not a sustained multi-update baseline. Export time combines checkpoint save and four exports: dynamic CPU, fixed GPU batches 512/256/32 at 48 tokens. Future engine-build cost is not captured by this single warm-start update. Treat these costs as opportunities, not established removable time.

Hardware context: Ryzen 3600X (6 cores/12 threads), RTX 3060 12 GB, WSL2 with about 15 GiB Linux RAM. Bound host-memory growth.

## Strict collaboration boundary

Another workstream owns GPU performance. This agent must:
- Run all Python validation with `CUDA_VISIBLE_DEVICES=''` and CPU execution providers explicitly selected.
- Never launch GPU training, TensorRT builds, GPU inference, GPU profiling, or device benchmarks. Do not assume hiding CUDA devices makes arbitrary native CUDA calls safe.
- Not edit `gpu.cpp`, `workers.*`, `net.hpp`, `play.cpp`, packing/search code, model architecture, or inference precision policy.
- Prefer new owner-local modules/tests. Coordinate before editing `loop.py`, `model.py`, or `test_loop.py`; preserve unrelated pending edits. Historical note: the GPU-overlap experiment that was uncommitted when this spec was written was subsequently archived and removed by user-approved cleanup; do not reintroduce its CLI or implementation while integrating the overhead work. See `docs/ASSESSMENT_AND_NEXT_STEPS.md`.
- Keep protocol/checkpoint/sample compatibility. No dependencies without approval; follow AGENTS.md if dependencies become necessary.
- Use a separate `build/<name>/` if native compilation becomes necessary; avoid rebuilding the other workstream's binary.
- State runtime/goal/log path before commands that may exceed one minute, run them in the background, preserve logs. Bound CPU parallelism (start at 2 threads), and coordinate longer CPU benchmarks to avoid contaminating GPU measurements.

## Priority A — incremental replay cache

Implement an in-memory, CPU-side rolling replay cache; keep the existing `read(paths)` API as the reference/fallback.

Requirements:
1. Decode each completed immutable sample file once while it remains in the active replay window. Do not use partial collection files.
2. Preserve ordered concatenation by supplied path and record order, every row, outcomes, token/policy placement, dtypes, and padding semantics exactly.
3. Retain data per file (or another simple bounded representation) and evict entries outside the current window. Avoid retaining large raw buffers through NumPy views accidentally.
4. Handle the maximum token length increasing or decreasing as files enter/leave. Do not silently fix a token cap, truncate states, or introduce a different training padding shape.
5. Resume cold from existing files. Correctness cannot depend on a persistent cache sidecar. Clear or fail on unexpected file modification; explicitly define the immutable-file contract and identity checks.
6. Do not change uniform row sampling, RNG consumption, replay window selection, training steps/batch, or bootstrap behavior.
7. Distinguish decode time from dense assembly time. An initial cache may still reassemble the dense tensors; optimize further only if measurements warrant it. Avoid introducing a complex GPU replay store in this workstream.
8. Keep memory bounded by the replay window plus a documented assembly allowance. Compare peak RSS against the original reader; don't buy a minor speed gain with unbounded duplicate dense replays.

Tests (CPU, synthetic binary fixtures): cache versus `read` exact tensor equality; append/update/evict; changing longest state; bootstrap and multiple files per iteration; repeated access doesn't re-decode; file mutation handling; cache lifetime and interrupted-file rejection behavior. Validate malformed/truncated inputs clearly if touching parser behavior. Preserve the legacy empty-input behavior or explicitly document/test any deliberate improvement.

## Priority B — avoid repeated graph export where safely possible

First measure CPU checkpoint-save, model snapshot/copy, graph export, and serialization separately using synthetic initialized weights of the actual 128/4/4 architecture. Existing run artifacts may be read, never changed.

Preferred investigation: export a graph template once per architecture/input signature, then instantiate it with current weights on subsequent updates. ONNX graph shape/signatures and FP16 input transport must remain unchanged.

This is conditional, not a mandate: optimized ONNX graphs can fold/transform parameters, including transposes, casts, or concatenations. Blind initializer-name replacement is NOT acceptable. If a small robust parameter mapping cannot be demonstrated, stop and report rather than shipping fragile weight patching.

Requirements if implemented:
1. Explicitly identify every mutable parameter's transformation/mapping. Reject unsupported graph/exporter signatures; retain the full-export fallback.
2. Reuse topology only, never stale weights. Validate all parameter updates, including embeddings, attention, feed-forward, norms, and heads.
3. Maintain dynamic CPU export and fixed FP16-input exports for each configured bank shape; no hardcoded 512/256/32 or architecture.
4. Preserve external-data behavior. Every output must be self-contained with its own correctly referenced weight data; never mutate a template or share writable weights across iteration directories.
5. Do not move the live training model off its device or alter its train/eval state as an export side effect. CPU snapshots may be made; GPU copy cost remains unverified here. Avoid excess simultaneous model copies.
6. Preserve checkpoints and optimizer/RNG state. Checkpoint saving remains mandatory every completed iteration; do not suppress it for a benchmark win.
7. Write complete model artifacts before publishing/returning them. A failed export must not look like a completed usable artifact. Preserve restart quarantine rules.
8. Record graph reuse/fallback and per-phase times. Report output hashes as provenance, not proof of predictive equivalence.

CPU validation:
- Compare fresh full exports and reused-template exports against PyTorch CPU for at least three distinct weight snapshots, including a real CPU optimizer update and targeted changes to individual parameter families.
- Test dynamic batches/token lengths and at least two fixed batch shapes; use small shapes for routine tests, representative shapes for bounded CPU benchmarks.
- ONNX checker plus ONNX Runtime CPU provider inference, finite values, value/logit closeness, legal-action ordering/masks, output shapes and input dtype. Declare tolerances and report maximum observed differences; no arbitrary tolerance relaxation to hide stale weights.
- Include incompatible signatures, missing/duplicate mappings, external-data relocation, and repeated exports to different new directories.
- Exact numeric GPU/TensorRT equivalence and engine-refit compatibility are explicitly NOT established by these tests.

## Integration and compatibility

Deliver A and B separately; A should not depend on B succeeding. Prefer opt-in rollout until reference-equivalence tests pass and CPU benchmarks show a benefit. Preserve normal bootstrap, resume, evaluation and model-bank selection behavior. Historical checkpoints without a `spec` or RNG state must remain supported as currently documented.

Production integration belongs in `loop.py`; keep the short benchmark harness consistent if enabled there. Never validate only a new isolated microbenchmark and claim the production loop has improved. Existing CLI defaults and resume compatibility must not silently change. New internal implementation choices must not make historical runs impossible to resume.

## Out of scope / deferred GPU lifecycle work

Do not implement a persistent native GPU service, TensorRT refitting, new native weight format, or asynchronous training/self-play here. ONNX graphs contain trained weights; retaining a process or engine alone does not update them. A future GPU-validated design must update ALL fixed-shape engines AND oversized-state CPU fallback to the same weight version, invalidate/recapture graphs as needed, and enforce a quiescent update boundary. Record interface suggestions only; don't build an unvalidated service around an assumed refit capability.

Likewise, no changed model, token cap, search budget, number of games, training recipe, or dropped long fights to manufacture throughput improvements.

## Deliverables and acceptance

1. Small independently reviewable implementation(s), owner-local tests, usage/fallback documentation.
2. CPU-only benchmark report under a NEW `scratch/nn-overhead/<tag>/` directory; no source-tree weights or generated caches. Synthetic deterministic inputs are preferred.
3. Compare baseline and candidate over at least five successive replay-window updates and five exports after the first template export. Report cold initialization separately, per-update times, sample sizes, observed min/median/max (not confidence intervals), peak RSS, and CPU/thread configuration.
4. Include correctness checks in the measured workflow; preserve raw commands/logs. Clearly separate graph construction savings from serialization/copy costs that remain.
5. Summarize what changed, measured savings, memory trade-offs, files touched, fallback behavior, and outstanding GPU validation. No claims of training strength or sustained end-to-end speedup.

Suggested initial acceptance bar: eliminate repeated replay decoding without input changes; demonstrate a clear CPU export-time reduction if graph-template reuse is feasible. A speculative exporter rewrite with no robust parameter mapping is worse than deferring B. Bring findings back before attempting GPU lifecycle integration.
