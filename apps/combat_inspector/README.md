# Combat inspector

Rerun one decision using an exported checkpoint's CPU network, independently of the training app.
Build with `make build`, then:

```python
from apps.combat_inspector import decision

result = decision('/path/to/checkpoint', fight['start'], moves=[], sims=64, rollout_mix=0.5)
if not result['over']:
    graph = result['graph']
    root = next(node for node in graph['nodes'] if node['observation'] == graph['root'])
    print(result['decision'])
    for edge in root['edges']:
        print(edge['description'], edge['visits'], edge['q'])
```

The checkpoint's `network.toml` selects the encoding and CPU artifact. Each call starts a fresh process
and loads the model; there is no session or model cache. `executable=...` selects another build.
Native failures raise `subprocess.CalledProcessError`, with the diagnostic in `stderr`.

`moves` contains previously chosen action bits; forced moves are automatic. The response contains `steps`
and `over`, plus `decision` and `graph` when there is a decision. Append `decision.action` to inspect the next one.
This reruns search; it does not retrieve a recorded training-time search or promise CPU/GPU numerical identity.

The graph contains shared observation nodes, not a tree. Each node reports `evaluation_value` (including any
rollout blend), `mean_value` over its visits, and edges with priors, visit counts, Q, Gumbel sigma and improved
policy. Q is the traversal-weighted mean of child means; it is null when unvisited. Node visits include its
initial evaluation; edge visits count traversals. Terminal nodes have no edges. Root edges also carry playable
`bits` and a `description`; other edges use public action keys. Observation IDs and public action keys are
64-bit decimal strings. `root_noise` follows root edge order.
