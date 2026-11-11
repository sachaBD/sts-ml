# sts_combat_rl

A Slay the Spire combat agent: Gumbel search plus a neural network, trained by expert iteration.
Built on the simulator in the sibling checkout `../lightspeed` (override with `STS_LIGHTSPEED_DIR`).

## Setup

```sh
make env          # .venv via uv
make build        # build/main, including the nn_play worker
make test         # native, Python and dashboard tests
```

## Run

See [`apps/gumbel_nn_combat_expert_iteration/README.md`](apps/gumbel_nn_combat_expert_iteration/README.md).

## Layout

- `apps/gumbel_nn_combat_expert_iteration/`: the training app and dashboard.
- [`apps/combat_inspector/`](apps/combat_inspector/README.md): Python API for inspecting a decision's search graph.
- `agents/combat/nn/`: the network (entity and inventory models), their encodings, and training.
- `agents/combat/player/`: the agent that plays fights, search + network (`nn_play`).
- `agents/combat/search/`: Gumbel search.
- `environments/combat/`: the rules of a fight as a player sees them, and loading fight starts.
- `infra/`: inference backends, worker plumbing and worker telemetry.
- `runs/`: the run launcher. Results go in `runs/` (not committed).
