# Worker-local scheduler (v1)

## Review entry points

- `workers.hpp`: `Simulation`, `WorkerConfig`, synchronous evaluation contract.
- `workers.cpp`: `SimWorker::run`, `Coordinator::run`, `run_workers` supervision.
- `play.cpp`: `CombatSimulation` adapter; existing `Game` search/simulator logic is unchanged.
- `gpu.cpp`: synchronous caller-owned mode for the coordinator (no additional dispatch thread).

## Flow / ownership

Each of N SimWorkers owns K slots, their simulator/search state, a ready-ID deque, and one mutex-protected mailbox with separate request/completion deques. Each deque is bounded by K. Workers claim new starts from the shared atomic start index. Maximum active games is N*K, not the batch size.

A slot is READY, WAITING_FOR_NN or FINISHED. It sends at most one immutable token view at a time. Every request increments its slot generation; responses contain slot/generation and predictions only. The coordinator gathers requests round-robin, runs one synchronous inference batch, and routes replies. Simulator states/trees never migrate. Owners alone advance/evaluate/finish their games. Finished output uses the existing serialized writer, not the coordinator.

Overflow is a fatal invariant violation, not a blocking producer operation. Request/completion capacity follows from the one-outstanding-request-per-slot rule. Coordinator and worker waits use notifications plus bounded timeout checks. No lock-free queues, affinity, stealing or adaptive capacity.

## Explicit configuration (legacy defaults retained for review)

Native:

```
nn_play play MODEL.onnx SAMPLES.bin --scheduler local --threads 9 --parallel 224 \
  --gpu-model MODEL-GPU.onnx --gpu-batch 512 --gpu-tokens 48 --gpu-wait-us 100 \
  --timeout-ms 60000 --shutdown-timeout-ms 5000
```

`--threads` counts SimWorkers; local adds ONE coordinator. Main only supervises; backend libraries may have their own internal threads. `--parallel` is slots per SimWorker. `--groups` is ignored by local. Batch shapes/bank remain explicit and same-weight; no truncation. Add `--gpu-bank SMALL256.onnx,SMALL32.onnx` if available. CPU-only mode also supports local.

Python loop: `--scheduler local --workers 9 --parallel 224 --gpu-batch 512 --gpu-bank 256,32 --gpu-wait-us 100` exports matching models automatically. No training launch implied. Default remains legacy `ordered` until review. Old `ordered`/`ready` modes remain for comparison.

## Failure behavior

Defaults: 60 s outstanding-request / worker-progress deadline, 5 s shutdown grace. Workers inspect pending requests at least every10ms when not inside simulator code. Main supervises per-worker and coordinator heartbeats, so stuck simulation/inference is also detected. Errors cancel collection and propagate; no timeout becomes a loss or implicit retry. Only genuinely finished games are recorded; a hard exit may still leave an incomplete binary tail. Quarantine the entire interrupted stage and never append on restart.

Slot storage outlives BOTH worker and coordinator threads during cancellation. A native GPU/simulator call cannot safely be interrupted in C++. If it fails to return during shutdown grace, supervisor logs and exits nonzero with `_Exit`, rather than hanging on join or freeing in-use inputs. This hard-exit path is tested in a subprocess. Engine startup occurs before collection supervision and is not covered by these request timeouts.

## Implementation / recovery log

- User approved the local-ownership design after the earlier throughput investigation; its pending work/artifacts are preserved. No new training authorized or launched.
- New files: workers.hpp/cpp, workers_test.cpp, opt-in test_workers_integration.py. Implementation and checks completed; see verified results below.
- Build commands: `cmake --build build/nn --target nn_play nn_workers_test --parallel 6`.
- Tests: `ctest --test-dir build/nn -R 'nn_workers_test|python.agents.combat.nn' --output-on-failure`.
- Planned checks: fake simulations with variable lengths/slot reuse/owner-thread assertions/capacity; backend and factory failures; prediction validation; cancellation lifetimes; uninterruptible backend watchdog; small frozen-model real-game CPU/GPU checks.
- Generated logs/artifacts for this implementation: `scratch/nn-local-workers/`.

## Verified results

- Release native worker tests pass: eight variable-length100-game stress runs;1/2/3 slots per worker; slot reuse, owner-thread assertions, queue/global capacity, empty jobs, idle workers, backend/factory failure, malformed predictions, cancellation input lifetime, stalled simulation suppression, and a hung-backend subprocess shutdown.
- Owner Python tests pass, including local scheduler CLI/bank export settings and legacy checkpoint continuation checks.
- Opt-in real-model checks pass:8 fixed CPU fights have identical wins/HP/decision counts versus legacy;32 complete GPU fights at64 simulations produce valid finite token/policy samples, normalized targets, exact fight IDs and matching sample/decision counts. These are correctness samples, not strength or speed estimates.
- AddressSanitizer + UndefinedBehaviorSanitizer (including normal-shutdown leak checks) pass on the native scheduler tests. ThreadSanitizer could not start on this WSL host (`unexpected memory mapping`); race-detector verification remains unavailable.
- Logs: `scratch/nn-local-workers/{build-reviewed.log,tests-reviewed.log,integration.log,asan.log,sanitizers.log}`. Real samples/commands/results: `scratch/nn-local-workers/integration-v1/`.
- No training launched. Local scheduler throughput at the intended9×224 configuration remains unmeasured. In particular, coordinator-owned packing is deliberately simple and may serialize input preparation with inference; do not assume the old group-scheduler throughput results transfer.
- Existing frozen-model benchmark harness now accepts `--scheduler local`. Any future comparison must match total application-thread budget, slot count, frozen weights, starts and batch-bank settings; use new tags and stable-host trials.

Reproduce the opt-in integration checks with explicit `NN_TEST_MODEL`, `NN_TEST_STARTS`, `NN_TEST_GPU`, `NN_TEST_GPU_BANK` paths and a NEW `NN_TEST_OUT` directory, then run `.venv/bin/python -m unittest agents.combat.nn.test_workers_integration -v`. Default CTest deliberately skips real-model tests rather than discovering recorded runs.
