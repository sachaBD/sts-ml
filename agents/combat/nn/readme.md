# Combat NN

Combat models, learning and native play. Training workflows belong to the app.

- [Run and configure training](../../../apps/gumbel_nn_combat_expert_iteration/README.md)
- [Repository setup and build](../../../README.md#setup)
- [Model contract](model.py) and colocated invariant tests
- [Worker boundaries](WORKERS.md)
- [Runtime metric definitions](METRICS.md)
- [Performance reports](docs/README.md) — historical measurements, not current defaults

## Explicit inventory model

`model.kind = "inventory_v1"` selects individual hand cards, off-hand card variants with
per-pile counts, player/monster effects with owner links, potions, relics and legal actions.
The shared full-attention transformer retains the existing policy, value and auxiliary heads.
Card identities are shared across hand, inventory and card-action tokens.

Use `dim = 192`, `layers = 4`, `heads = 6`, `value = "outcome"`, `auxiliary = true`
for the discussed topology. Omit `gpu_attended` and `gpu_wide_attended`: all tokens attend;
`gpu_tokens` and optional `gpu_wide_tokens` bound the exported input shapes.
Training selects the native encoder automatically. Standalone native play/import/check
and explanation requests select `inventory_v1` through `--encoding` or the `encoding` request field.
This is a new checkpoint architecture and input schema, not a compatible warm start from
`entity`/`transformer` weights or replay. Existing model kinds follow main's version-2
target-linked encoding; checkpoints without that encoding version are rejected.
