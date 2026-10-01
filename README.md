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
./apps/bootstrap/run.sh apps/bootstrap/act1.toml        # teacher self-play -> combat_v3
./apps/value_train/run.sh apps/value_train/slime.toml   # train a value net -> value_net_v1
./apps/value_play/run.sh apps/value_play/slime.toml     # play fights with a value net
./apps/gauntlet/run.sh apps/gauntlet/act1.toml         # gauntlet card-reward evaluation
```

Query run outputs with DuckDB through `sts_combat_rl.query`.

## Layout

- `combat/` wraps `sts_lightspeed` into an environment (legal actions, steps) and the `combat_v3` state encoding.
- `agents/` holds the search teacher (`teacher_search`, `teacher_leaves`).
- `topology/` holds the value net topologies (frozen once used; see its README): C++ inference here, PyTorch in `python/sts_combat_rl/topology/`.
- `scenarios/` builds starting states: the Act 1 run replay plus fixed fights used by tests and benches.
- `apps/` holds the runnable jobs (C++ worker + Python driver + configs).
- `python/sts_combat_rl/` holds the topologies, training, the run launcher and the query layer.
- `experiments/` holds dated experiment write-ups with their configs and analysis scripts.
