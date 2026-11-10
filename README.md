# sts_combat_rl

A Slay the Spire combat agent: Gumbel search plus a neural network, trained by expert iteration.
Built on the simulator in the sibling checkout `../lightspeed` (override with `STS_LIGHTSPEED_DIR`).

## Setup

```sh
make env-onnx     # .venv via uv
make build        # build/main, including the nn_play worker
make test         # native, Python and dashboard tests
```

## Run

See [`apps/gumbel_nn_combat_expert_iteration/README.md`](apps/gumbel_nn_combat_expert_iteration/README.md).

## Layout

- `apps/gumbel_nn_combat_expert_iteration/`: the training app and dashboard.
- `agents/combat/nn/`: the network, training, export, and the native `nn_play` worker.
- `agents/combat/search/`: Gumbel search.
- `environments/combat/`: fight state, actions, and encodings.
- `infra/`: inference backends and worker plumbing.
- `monitoring/`, `runs/`: telemetry and the run launcher. Results go in `runs/` (not committed).
