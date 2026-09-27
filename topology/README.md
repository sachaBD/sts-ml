# topology/

Value net topologies: untrained network shapes. Trained weights are runs (`runs/schema=value_net_v1/`).

- One kind per file: `python/sts_combat_rl/topology/<kind>.py`, mirrored for inference in `value_net.cpp`.
- Frozen once used: after a run uses a kind, its behaviour never changes. A new idea is a new kind (`deep_sets_v3`).
- Nothing implicit: an architecture is `kind` plus every constructor argument; checkpoints store it in full.
- Checkpoints from before kinds existed are `deep_sets_v1`.
- Kinds read an encoding version: `deep_sets_v1` / `deep_sets_v2` read v3; `deep_sets_v3` reads v4 (`combat/encoding_v4.cpp`: player and monster statuses, previous monster move, potion and relic tokens; Ironclad act 1, with `EXTEND HERE` marks) and outputs the terminal score formula over win / HP / potions-kept heads.
