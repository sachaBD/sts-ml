# Combat episodes on shared infrastructure

`play.cpp::CombatEpisode` implements `infra::Episode`: it owns a fight and Gumbel
search, advances until inference is needed, consumes a prediction, and writes the
finished game's samples and result under the output lock. Only immutable
`infra::Tokens` and `infra::Prediction` cross the worker boundary.

`infra/workers/actors.hpp` owns actors, slots, mailboxes, batching, supervision,
timeouts and cancellation. `infra/inference/backend.hpp` owns model loading,
packing and synchronous CPU/GPU inference. Neither layer imports combat code.
See [infra's contract](../../../infra/workers/readme.md).

```
nn_play play MODEL.onnx SAMPLES.bin --threads 9 --parallel 224 \
  --gpu-model MODEL-GPU.onnx --gpu-batch 512 --gpu-wait-us 100 \
  --gpu-bank SMALL256.onnx,SMALL32.onnx
```

`--threads` counts actors, `--parallel` is slots per actor, and `--gpu-batch`
bounds the inference-server batch. Backend shapes/width come from ONNX metadata;
the combat adapter checks width against its encoder. The loop's `--gpu-tokens`
controls export only. No scheduler/groups options or old evaluator remain.

Failures abort collection, never become losses. Defaults are a 60-second stall
deadline and 5-second shutdown grace. Archive interrupted stages; never append
partial replay. A poisoned CUDA context exits without teardown for the app's
bounded process supervisor to handle.

Training always uses agent-owned disk-indexed, `Replay`, backed by
`infra.Reader` and `infra.prefetch`. Only length-bounded FP32 minibatches enter
RAM/GPU. Sampling, binary format, targets and optimizer remain agent-owned.

Verification: `infra_actors_test` covers scheduling/failure contracts;
`test_workers_integration.py` checks complete combat results, sample validity and
CPU slot-count parity with explicitly supplied models/starts. `test_replay.py`
covers indexing, sampling and optimizer updates.
