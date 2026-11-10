# Combat player

The agent that plays fights, run by `nn_play` (`play.cpp`) and launched from Python by `play.py`. It knows two interfaces:

- `search::SearchFactory` (`search/search.hpp`): a fresh search per decision. `search::SearchConfig` (`search/config.hpp`)
  owns the search flags and which search runs; the player never names it.
- `nn::NetworkEvaluator` (`nn/evaluator.hpp`): a search query becomes network tokens; a prediction becomes an evaluation.

`Decision` joins one search to the network; infra batches the network calls of every decision in flight.

## Combat episodes on shared infrastructure

`play.cpp::CombatEpisode` implements `infra::Episode`: it owns a fight and its search, advances until inference is needed, consumes a prediction, and writes the
finished game's samples and result under the output lock. Only immutable
`infra::Tokens` and `infra::Prediction` cross the worker boundary.

`infra/workers/actors.hpp` owns actors, slots, mailboxes, batching, supervision,
timeouts and cancellation. `infra/inference/backend.hpp` owns model loading,
packing and synchronous CPU/GPU inference. Neither layer imports combat code.
See [infra's contract](../../../infra/workers/readme.md).

```
nn_play play MODELS SAMPLES.bin --threads 9 --parallel 224 --gpu-wait-us 100 --timeout-ms 60000 \
  --shutdown-timeout-ms 5000 --telemetry telemetry.jsonl --sims 64 --particles 8 --gumbel-scale 1 --rollout-mix 0
```

Every option is required. `MODELS` is an exported model directory: its `network.json`, written by
[`pytorch_to_onnx.py`](../nn/pytorch_to_onnx.py), names the encoding and the GPU deployments, smallest token
capacity first, so nothing about the network is passed as a flag. Each input goes to the smallest deployment
that fits it; an input longer than every deployment is an error. `--threads` counts actors and `--parallel`
is slots per actor. The player checks the models' token width against its encoder.

Failures abort collection, never become losses: a stall past `--timeout-ms` is a failure, and shutdown
waits at most `--shutdown-timeout-ms`. Archive interrupted stages; never append
partial replay. A poisoned CUDA context exits without teardown for the app's
bounded process supervisor to handle.

Combat sample ranges let replay identify combats without inferring boundaries
from journal order. Keep the result journal beside its sample file. The two files
are not crash-atomic; trailing sample bytes alone do not prove a completed combat.

Training always uses agent-owned disk-indexed, `Replay`, backed by
`infra.Reader` and `infra.prefetch`. Only length-bounded FP32 minibatches enter
RAM/GPU. Sampling, binary format, targets and optimizer remain agent-owned.

Verification: `infra_actors_test` covers scheduling/failure contracts;
`test_workers_integration.py` checks complete combat results, sample validity and
CPU slot-count parity with explicitly supplied models/starts. `../nn/training/test_replay.py`
covers indexing, sampling and optimizer updates.
