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

## Lab: drive a search yourself

`apps.combat_inspector.lab` keeps searches open in one `combat_lab` process and lets you pick every
simulation's root move and world (see its docstring and `lab.cpp` for the protocol):

```python
from apps.combat_inspector.lab import Lab

with Lab('/path/to/checkpoint') as lab:
    tree = lab.tree(fight['start'], fight['moves'])
    tree.simulate([(0, 0), (1, 0)])        # one wave: (move, world) pairs
    tree.root['moves'][0]['worlds']        # per world: visits and that world's own value of the move
```

Worlds are an endless indexed stream per draw (draw 0 is play's), so a policy can add worlds as it goes.
`lab.simulate({tree: picks, ...})` steps many trees in one network batch.
