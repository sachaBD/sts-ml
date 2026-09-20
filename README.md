# sts_combat_rl

Small experiments toward a learned Slay the Spire combat agent, built on
[`sts_lightspeed`](https://github.com/gamerpuppy/sts_lightspeed).

The first milestone is intentionally small: an Ascension 1 Ironclad starter
deck fighting Jaw Worm, controlled by a random policy. The project code targets
C++23. The agent only receives a public observation and legal actions; the
simulator's hidden state remains behind the environment boundary.

## Build and run

```sh
git submodule update --init --recursive
make jaw_worm
```

Use a particular episode seed with `make jaw_worm SEED=42`. The Makefile is a
small convenience wrapper around CMake; `make build` only builds, and
`make clean` removes the build directory.

Run the information-set MCTS agent with:

```sh
make jaw_worm_mcts
make jaw_worm_mcts SEED=42 SIMULATIONS=2000
```

### PyTorch value-model smoke test

Create the project-local virtual environment, build the C++ record generator, and run one value-model forward pass against Parquet records:

```sh
make smoke
```

This prints the public JSON encoding of the record, model topology, tensor shapes, and one untrained value. The tensor adapter intentionally ignores legal actions.

MCTS uses the random agent for terminal rollouts. Each simulation independently
reshuffles the unknown remainder of the draw pile and resamples future RNG, so
the search cannot condition its root choice on the simulator's true hidden
order. This is an initial determinization-based information-set search, not a
perfect-information oracle.

### MCTS Parquet pilot

Generate a value-only Slime Boss shard (the C++ generator streams MessagePack; Python writes Parquet):

```sh
make dataset SIMULATIONS=2000
# or: PYTHONPATH=python .venv/bin/python -m sts_combat_rl generate dataset --seed-count 32 --simulations 2000
```

The output directory contains `mcts_slime_v2.parquet` and `manifest.toml`. Inspect rows as JSON with:

```sh
PYTHONPATH=python .venv/bin/python -m sts_combat_rl inspect <path-to-parquet> --row 0
```

## Layout

- `agents/` contains a uniformly random policy and an information-set MCTS agent.
- `combat/` adapts `sts_lightspeed` into observations, legal actions, and steps.
- `scenarios/` constructs reproducible combat starting states.
- `apps/` contains runnable experiments.
- `sts_lightspeed/` is the pinned upstream simulator submodule.

The observation is deliberately a small human-readable view, not the final ML
state encoding. We can evolve that representation once the environment loop is
working and measurable.
