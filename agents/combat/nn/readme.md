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

Omit `gpu_attended` and `gpu_wide_attended`: all tokens attend; `gpu_tokens` and optional
`gpu_wide_tokens` bound the exported input shapes. Oversized states use the wider model or CPU,
not truncated inputs.
Training selects the native encoder automatically. Standalone native play/import/check
and explanation requests select `inventory_v1` through `--encoding` or the `encoding` request field.
This is a new checkpoint architecture and input schema, not a compatible warm start from
`entity`/`transformer` weights or replay. Existing model kinds retain their version-2 encoding.

The native layout is defined in `inventory_tokens.cpp`; Python reads it in
`inventory_transformer.py`. Rows retain the shared 68-float transport: kind/identity fields,
type-specific features, then one-based owner/source/target references (zero means absent).
The context marker distinguishes this schema from legacy replay. Hand cards remain individual;
off-hand copies with the same encoded modifications share a row with draw/discard/exhaust counts.
