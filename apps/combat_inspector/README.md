# Combat inspector

Inspect why a combat agent chose an action by replaying a fight and examining its search graph,
independently of training.

Build with `make build`, then use from Python:

```python
from apps.combat_inspector import decision, decisions, request

result = decision('/path/to/checkpoint', fight['start'])
results = decisions('/path/to/checkpoint', [request(f['start'], sims=256) for f in fights])
```

`decisions` searches every request in one `combat_explain` process, which batches network calls across
them on the GPU deployments of the checkpoint's `network.toml` (`cpu=True` for the CPU only). Each response
reports its decision, search graph and `network_requests`. `request` documents the fields of a request line.

Inspection reruns search; it does not retrieve the original training-time search.
