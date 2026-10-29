# Workers and resources

Goal: every CPU worker exists to keep the GPU saturated (NN ceiling ~110k evals/s), in play and in train.

One iteration = **play** phase, then **train** phase. They never overlap, so the one GPU is
used by the InferenceServer during play and by the Learner during train.

## Resources

| Resource | Size | Used for |
|---|---|---|
| CPU | Ryzen 5 3600X: 6 cores / 12 SMT threads | Actors (play), Readers (train) |
| GPU | 1 card, **shared** | InferenceServer (play), Learner (train) |
| RAM | ~15.6 GiB (WSL cap) | game/search state, batch queue, page cache |
| Disk | replay window ~20 GiB (10 iters x ~2 GiB) | samples written by Actors, read by Readers; model files |

## Workers

| Worker | Count | Process | Phase | Does |
|---|---|---|---|---|
| Loop | 1 | python `loop` | both | starts play, runs train, exports models, prunes old replay |
| Actor | 9 (`--workers`) | native `nn_play` | play | plays K games each (`--parallel`): search + simulator, requests NN evals, writes samples |
| InferenceServer | 1 | native `nn_play` | play | batches all Actors' requests, runs the GPU, routes replies |
| Reader | 4 threads | python `loop` | train | reads sampled rows with coarse `pread` tasks; Replay builds padded batches |
| Learner | 1 | python `loop` | train | takes ready batches, runs the GPU training step |

## Diagram

```
                           ┌──────────┐
                           │   Loop   │  play phase -> train phase -> repeat
                           └────┬─────┘
             ┌──────────────────┴───────────────────┐
             ▼ PLAY  (nn_play process)              ▼ TRAIN  (loop process)
 ┌──────────────────────┐                ┌──────────────────────┐
 │ Actor x9  (CPU)      │                │ Reader x4   (CPU)    │
 │ games + search       │                │ pread rows -> batch  │
 └───┬─────────▲────┬───┘                └───▲─────────────┬────┘
     │ mailbox │    │ finished games         │ random rows  │ ready batches
     ▼         │    │                        │              ▼
 ┌─────────────┴─┐  │   ┌──────────────┐     │      ┌───────────────┐
 │InferenceServer│  └──►│ DISK replay  ├─────┘      │ batch queue   │
 │ batch requests│      │ samples-*.bin│            │ (2-3 batches) │
 └──────┬────────┘      └──────────────┘            └───────┬───────┘
        │                                                   ▼
        │                                              ┌─────────┐
        │ inference                                    │ Learner │
        │                                              └────┬────┘
        │                   ┌──────────────┐                │ fwd/bwd
        └──────────────────►│ GPU (shared) │◄───────────────┘
                            └──────────────┘
```

## Communication

| From -> To | Mechanism |
|---|---|
| Loop -> nn_play | spawn subprocess; model paths as args; starts on stdin |
| nn_play -> Loop | one JSON result line per finished game on stdout; errors to `worker.log` |
| Actor <-> InferenceServer | in-process: per-Actor mailbox (mutex, request/completion deques bounded by K slots) + condition-variable wakeups |
| Actor -> Disk | shared mutex-guarded `ofstream` appending `samples-0.bin` |
| Reader -> Disk | `pread` exact row bytes |
| Reader -> Learner | in-process bounded `queue.Queue` of padded batches |
| Learner -> next play | `model.onnx` files on disk |
| Loop -> dashboard | `status.json`, `log.jsonl` |

## Rules

- Readers stay ahead of the Learner by the queue depth; the GPU should never wait on disk.
- Read exact row bytes (`pread`), never `mmap` random rows: WSL read-ahead is 8 MiB per fault.
- Schedule at most eight chunks per reader thread per batch, not one task per row.
  Chunks read rows sequentially; the returned rows preserve request order, including duplicates.
- The replay window does not need to fit in RAM.

## Code

| Piece | Where |
|---|---|
| Actor, InferenceServer, supervision | `infra/workers/actors.hpp` |
| Reader, prefetch queue | `infra/workers/reader.py` |
| Backend interface, CPU/GPU backends | `infra/inference/backend.hpp` |
| Component benchmarks | `python -m infra.benchmarks MODEL_DIR REPLAY_FILE...` (`infra/benchmarks/`) |
| Learner (losses, optimizer step) | `agents/combat/nn/train.py`: agent-specific, stays with the agent |
| Episode for combat (search + simulator) | `agents/combat/nn/play.cpp`: agent-specific |

`infra` knows nothing about Slay the Spire: workers depend on inference, and neither imports `agents/`, `environments/` or `apps/`.
