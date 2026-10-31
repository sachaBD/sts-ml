# NN100 simulation sweep

Approved: 16, 32, 64 and 128 simulations, once each on the same 500 starts (50/deck).
Checkpoint 132, NN-only leaves, eight particles, zero noise; frozen worker and input
hashes from `../2026-10-31-combat-leaf-mix/protocol.toml`. No training or baseline replay.

Reuse NN100/256: 298/500 wins, 113.076 s. Baseline outcomes are at
`/home/sborowsk/project/sts/worktrees/combat-leaf-mix-benchmark/runs/benchmark/leaf-mix-132-approved/nn100/results.jsonl`,
SHA-256 `72eae4bd11baad16789eb3ab23a418700daae150faac74e343ef4b066308dc56`.

```sh
/home/sborowsk/project/sts/combat_rl/.venv/bin/python \
  experiments/2026-10-31-combat-leaf-mix/benchmark.py --sim-sweep \
  --run runs/benchmark/sim-sweep-132-approved
```

Run sequentially without other GPU work. Record paired differences vs 256, per-deck
wins and process wall time. One timing observation/budget; training-monitored starts,
not held out. Zero simulations is excluded because the frozen worker rejects it.
