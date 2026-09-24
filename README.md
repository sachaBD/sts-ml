# sts_combat_rl

Small experiments toward a learned Slay the Spire combat agent, built on
[`sts_lightspeed`](https://github.com/gamerpuppy/sts_lightspeed), via the shared
sibling checkout `../sts_lightspeed` (the `sts_ml` fork; override with
`STS_LIGHTSPEED_DIR`).

The current milestone is an AlphaZero-style combat prototype for an Ascension
1 Ironclad fight against Slime Boss. A PyTorch Deep Sets value model trained
from MCTS records now guides the C++ search. The project targets C++23. The
agent only receives public observations and legal actions; hidden simulator
state remains behind the environment boundary.

## Build and run

```sh
# requires ../sts_lightspeed (shared simulator checkout)
make jaw_worm
```

Use a particular episode seed with `make jaw_worm SEED=42`. The Makefile is a
small convenience wrapper around CMake; `make build` only builds, and
`make clean` removes `build/main`. All build dirs live under `build/<name>/`.

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

### Bootstrap value training

Train the batched, permutation-invariant Deep Sets value model on an episode-level held-out split (CPU
single-threaded). Configs live in `apps/value_train/`; the training rows are a SQL query
(`sts_combat_rl.query`, DuckDB over `runs/`):

```sh
./apps/value_train/run.sh apps/value_train/slime.toml [--scratch]
```

The checkpoint and adjacent JSON record exact episode split IDs, the data query and its source runs,
configuration, and metrics.

### Neural-guided gameplay evaluation

Run matched Slime Boss rollout and neural-leaf MCTS games (JSONL per episode plus aggregates):

```sh
PYTHONPATH=python .venv/bin/python -m sts_combat_rl.training.evaluate_gameplay \
  runs/slime-v2-value-first/value_checkpoint.pt \
  runs/gameplay.jsonl \
  --first-seed 2001 --count 200 --simulations 100
```

The neural controller keeps the checkpoint loaded once and communicates with one C++ child per episode over framed MessagePack.

The first held-out gameplay evaluation won 200/200 unseen fights. Detailed
configuration, metrics, hashes, and limitations are recorded in
[`slop_docs/slime/results_v1.md`](slop_docs/slime/results_v1.md).

The baseline MCTS uses the random agent for terminal rollouts; neural MCTS uses
the learned value for newly expanded nonterminal leaves and exact returns for
terminal leaves. Each simulation independently reshuffles the unknown remainder
of the draw pile and resamples future RNG, so search cannot condition its root
choice on the simulator's true hidden order. This is an initial
determinization-based information-set search, not a perfect-information oracle.

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
- `../sts_lightspeed` is the shared simulator, also used by `../sts_ml`.

The observation is deliberately a small human-readable view, not the final ML
state encoding. We can evolve that representation once the environment loop is
working and measurable.
