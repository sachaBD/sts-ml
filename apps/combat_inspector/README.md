# Combat inspector

Inspect why a combat agent chose an action by replaying a fight, running scheduled Gumbel searches, and steering
simulations on the same search graph. Inspection reruns search; it does not retrieve training-time searches.

All backends share one C++ engine (`inspection.hpp` / `inspection.cpp`) and public board representation
(`environments/combat/view.hpp`):

- **`combat_explain`** — offline GPU/CPU batch driver (`infra::run_actors`) for searching many decisions at once.
- **`combat_lab`** — persistent native GPU/CPU `Session` over JSONL stdin/stdout (`open`, `simulate`, `graph`, `state`, `close`).
- **`wasm/bridge.cpp`** — persistent browser WASM `Session` over Emscripten + ONNX Runtime Web, exposing the exact same
  `session_json` command protocol as `combat_lab` (with `init` to select the checkpoint's encoding and `start_json` to keep
  64-bit RNG seeds intact across JavaScript).

## Native / Python

Build with `make build`, then use from Python:

```python
from apps.combat_inspector import decision, decisions, request
from apps.combat_inspector.lab import Lab

# Offline batch inspection on GPU (set views=True for per-node public board views):
result = decision('/path/to/checkpoint', fight['start'], views=True)
results = decisions('/path/to/checkpoint', [request(f['start'], sims=256) for f in fights])

# Interactive session: auto-search and steer on the same live graph:
with Lab('/path/to/checkpoint') as lab:
    tree = lab.tree(fight['start'], search=dict(sims=64, particles=8, gumbel_scale=0, rollout_mix=0))
    tree.simulate([([0], 0, False), ([1], 0, True)])  # (move_or_path, world[, descend])
    tree.root['moves'][0]['worlds']                   # per-world visits and expectimax value
    tree.state([0], 0, worlds=8)['reaching']          # worlds 0..7 reaching that node
```

Observation and action hashes cross JSON as decimal strings so JavaScript cannot lose 64-bit precision.

## Browser WASM backend

With `.venv` prepared (`make env`) and Emscripten activated:

```sh
source /path/to/emsdk/emsdk_env.sh
make pages-wasm
node apps/combat_inspector/wasm/test_wasm.mjs
```
