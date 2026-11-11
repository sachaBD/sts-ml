# Combat inspector

Python API for rerunning one Gumbel decision and inspecting its search graph. No training app, run layout,
HTTP server or browser required. Inference uses the exported dynamic-shape CPU ONNX model.

Build with `make build` (or build target `combat_explain` after CMake configuration). This uses the same
inference build prerequisites as `nn_play`.

```python
import json
import tomllib
from pathlib import Path
from apps.combat_inspector import Inspector

checkpoint = Path('/path/to/checkpoint')
manifest = tomllib.loads((checkpoint / 'network.toml').read_text())
with open('/path/to/fights.jsonl') as fights:
    fight = json.loads(next(fights))

with Inspector() as inspector:
    result = inspector.decision(
        model=checkpoint / manifest['cpu']['file'],
        encoding=manifest['encoding'],
        start=fight['start'], moves=[],
        sims=64, particles=8, gumbel_scale=0, rollout_mix=0.5,
    )
    if not result['over']:
        nodes = {node['observation']: node for node in result['tree']['nodes']}
        root = nodes[result['tree']['root']]
        print('chosen:', result['decision'])
        for edge in root['edges']:
            print(edge['description'], edge['visits'], edge['q'], edge['prior'])
```

Pass chosen `decision.action` values back in `moves` to inspect later decisions. Forced actions are played
automatically, as in `nn_play`. An illegal move or invalid configuration raises `ValueError`.
`Inspector(executable=...)` selects a different build. Use the context manager to close the native process.
Calls on one inspector are serialized; the process caches the last model by path. Use immutable checkpoint
files, or close the inspector before inspecting an overwritten model.

## Response

- `steps`, `over`, `observation`: the replayed fight position. An over fight has no decision or tree.
- `decision`: chosen action bits and estimated value.
- `tree.root`, `tree.gumbel`, `tree.nodes`: root observation, root noise (edge order), and nodes sorted by observation.
- Node: `observation`, `terminal`, leaf `value` (after rollout blending), `network` (nonterminal only),
  `mean_value` (all simulations through the node), `visits`, `edges`.
- Edge: public `action` key, representative query's `bits` and `description`, `logit`, `prior`, `sigma`,
  `improved` policy probability, `visits`, `q` (null if unvisited), and `children` with observation and traversal count.

Despite the `tree` field name, this is a **graph**: multiple edges can reach the same observation.
Edge Q is the traversal-count-weighted mean of its children's current mean values, matching this search's
selection rule—not an independent historical edge accumulator. Node visits include its initial evaluation;
edge visits count traversals. Terminal nodes have no edges or network prediction.

IDs and public action keys are decimal strings to preserve all 64 bits. Action bits are integers and are valid
in the representative queried state; only root action bits should be used to extend the replayed fight.
This reruns search with the current implementation and CPU inference; it does not recover an original
training-time search or guarantee bitwise parity with GPU inference.

The executable also accepts one JSON request per stdin line and emits one JSON response per stdout line.
Requests contain `model`, `encoding`, `start`, `moves`, and `search` (`sims`, `particles`, `gumbel_scale`,
`rollout_mix`); native errors are returned as `{"error": "..."}` without ending the process.
