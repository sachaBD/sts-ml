# Combat NN

The network: two model types, each with its own input encoding, plus training.

- `model.py`: shared base class and `ModelSpec` (`kind = "entity"` or `"inventory_v1"`).
- `features.*`: what a player can see of a fight, as numbers; both encodings are built from it.
- `tokens.*`: the shared token format; `tokens.cpp` picks the encoding.
- `layout.toml`: every encoding's token layout and the sample record, the one source for both languages. C++ gets it as a header
  generated at build time; Python reads it (`layout.py`).
- `entity/`: the entity model and its encoding.
- `inventory/`: the inventory_v1 model and its encoding. It needs its own weights and replay.
- `pytorch_to_onnx.py`: writes the trained network as a model directory the player loads: the ONNX files
  and `network.toml`, which describes them (read by `network.cpp`).
- `training/`: replay, losses, the learner, telemetry.

The player that uses the network to play fights is in [`../player`](../player). Run training through
[the app](../../../apps/gumbel_nn_combat_expert_iteration/README.md).
