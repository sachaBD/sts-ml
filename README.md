# sts_combat_rl

Learned value models for Slay the Spire (Ironclad, Act 1) combat, built on
[`sts_lightspeed`](https://github.com/gamerpuppy/sts_lightspeed) via the shared sibling checkout
`../sts_lightspeed` (the `sts_ml` fork; override with `STS_LIGHTSPEED_DIR`). C++23 for the simulator,
search and native value net; Python (PyTorch, DuckDB) for training and analysis.

The loop: a search teacher plays Act 1 fights and records every decision (`combat_v3` rows), a
permutation-invariant Deep Sets value net is trained on those rows, and the net then replaces the
teacher's leaf evaluation for the next generation of self-play.

## Setup

```sh
make python-env   # .venv with requirements.txt
make build        # build/main (all build dirs live under build/<name>/)
make test         # ctest + Python unittests
```

## Running

Every job is a run with a TOML config, launched through `apps/<app>/run.sh` and written to
`runs/schema=<schema>/date=<date>/id=<id>/` (see [`runs/README.md`](runs/README.md)). Add `--scratch`
for smoke runs.

```sh
./apps/bootstrap/run.sh apps/bootstrap/config/act1.toml        # teacher self-play -> combat_v3
./apps/value_train/run.sh apps/value_train/config/slime.toml   # train a value net -> value_net_v1
./apps/value_play/run.sh apps/value_play/config/slime.toml     # play fights with a value net
./apps/gauntlet/run.sh apps/gauntlet/config/act1.toml         # gauntlet card-reward evaluation
```

Query run outputs with DuckDB through `runs.query`.

## Layout

Code is organized by purpose, not language:

- `environments/combat/`: fight state, legal actions, stepping, encodings/schema, and starting-state scenarios.
- `environments/overworld/`: persistent game state, legal after-states, replay, and real/injected combat orchestration.
- `agents/combat/`: search teacher and value/policy models, including their learning and native inference.
- `agents/overworld/`: learned after-state choices and model-based card-choice search, including agent-owned learning.
- `models/combat_outcome/`: shared pre-combat predictor used by search, evaluation, and the GUI; not an agent.
- [`apps/`](apps/INDEX.md): executable workflows, drivers, workers, and `config/` examples.
- `gui/`: local and published model explorer.
- `runs/`: launcher/query/compaction source plus durable generated results; `scratch/` holds disposable work.
- `experiments/`: experiment configs, analysis scripts, plans, and reports.
- `docs/`: documentation and clearly identified speculative research notes.
- `build/<name>/`: generated native builds.

**Environments own what can happen; agents own what to choose and how it is learned; apps execute workflows.**
Tests live beside their owners. `make test` runs native and Python checks through CTest;
`ctest --test-dir build/main -L python --output-on-failure` selects Python checks only.
Test registration does not crawl recorded run directories.

Model kinds, checkpoint dictionaries, architecture-spec contents, and native weight formats are unchanged by
this layout. Historical recorded commands in completed runs are left untouched. The previously active
run-RL v4 materials live in `experiments/run-rl-v4/`.
