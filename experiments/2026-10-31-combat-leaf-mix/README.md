# Checkpoint-132 leaf-mix sweep

**Status:** awaiting execution approval. No new fights run.

Compare **75% NN / 25% rollout** and **100% NN** against cached **50/50** results.
Use checkpoint 132 on the same 500 frozen starts: 50 per training deck.
Keep 256 simulations, eight particles and zero exploration noise. Run each variant once,
sequentially; no training, export or baseline replay.

## Inputs

`protocol.toml` pins local paths and SHA-256 hashes for the checkpoint, ONNX exports,
original worker, starts and cached outcomes. Existing totals:

| Reference | Wins / 500 |
|---|---:|
| NN50, checkpoint 132 | 288 |
| Historical PUCT | 237 |
| Historical MCTS20k | 175 |

## Usage

From this worktree, using the existing environment:

```sh
PY=/home/sborowsk/project/sts/combat_rl/.venv/bin/python
EXP=experiments/2026-10-31-combat-leaf-mix
$PY "$EXP/benchmark.py"  # read-only hashes and fight-ID checks; passes locally

# Only after execution approval; output directory must be new:
$PY "$EXP/benchmark.py" --run --out runs/benchmark/leaf-mix-132-ATTEMPT
```

Retain raw outcomes, commands, provenance, logs and process wall time in `runs/`.
No automatic retry or cleanup. After execution, record overall/per-deck results and
paired differences here using existing app statistics; no separate report generator.

## Essential caveats

- Use the pinned historical worker: PR #18 changed rollout blending and RNG streams.
  Exact historical dirty source/build settings were not archived; only the binary is pinned.
- At 100% NN the old worker skips rollout RNG draws, changing later search samples too.
- These decks and starts were monitored during training, not held out. Historical PUCT/MCTS
  simulator differences prevent a controlled algorithm comparison.
- Measure whole native-process wall time, including setup. One run/variant gives no timing
  uncertainty estimate; cached NN50 timing is not a same-session speedup control.
- Preflight passes; native execution remains unverified pending approval. Avoid other GPU work.
