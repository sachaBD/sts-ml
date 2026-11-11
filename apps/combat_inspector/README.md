# Combat inspector

Inspect why a combat agent chose an action by replaying a fight and examining its search graph,
independently of training.

Build with `make build`, then use from Python:

```python
from apps.combat_inspector import decision

result = decision('/path/to/checkpoint', fight['start'])
```

Inspection reruns search; it does not retrieve the original training-time search.
