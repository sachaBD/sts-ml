# Combat NN

Combat models, learning and native play. Training workflows belong to the app.

- [Run and configure training](../../../apps/gumbel_nn_combat_expert_iteration/README.md)
- [Repository setup and build](../../../README.md#setup)
- [Model contract](model.py) and colocated invariant tests
- [Worker boundaries](WORKERS.md)
- [Runtime metric definitions](METRICS.md)
- [Performance reports](docs/README.md) — historical measurements, not current defaults

## Inventory model

`model.kind = "inventory_v1"` requires its own weights and replay. All entities attend:
set `gpu_tokens` / `gpu_wide_tokens`, not pooled `gpu_attended` / `gpu_wide_attended`.
Training selects the encoder; standalone native play/import/check use `--encoding inventory_v1`,
and explanation requests use `"encoding": "inventory_v1"`.
