# Combat NN

The network: two model types, each with its own input encoding, plus training.

- `model.py`: shared base class and `ModelSpec` (`kind = "entity"` or `"inventory_v1"`).
- `features.*`: what a player can see of a fight, as numbers; both encodings are built from it.
- `tokens.*`: the shared token format; `tokens.cpp` picks the encoding.
- `entity/`: the entity model and its encoding.
- `inventory/`: the inventory_v1 model and its encoding. It needs its own weights and replay. All entities
  attend: set `gpu_tokens` / `gpu_wide_tokens`, not `gpu_attended` / `gpu_wide_attended`.
- `training/`: replay, losses, the learner, telemetry, and ONNX export for the player.

The player that uses the network to play fights is in [`../player`](../player). Run training through
[the app](../../../apps/gumbel_nn_combat_expert_iteration/README.md).
