# Overworld value agent

Owns `run_policy_v1`/`run_policy_v2` and the new `run_policy_v3_1`/`run_policy_v3_2` architectures, model-input encoding, after-state policy, TD(lambda) targets, checkpoint loading, and learning.

- [Map encoder explainer](../../../docs/research/topology/overworld-map.md)
- [v3 observations, variants, training and Heart caveats](../../../docs/research/topology/overworld-v3.md)

- `core.py`: model construction, batching, targets, and checkpoint/log helpers.
- `policy.py`: chooses cards and other overworld decisions, including sampled-lookahead evaluation.
- `graph_encoding.py`, `run_policy_v3.py`: versioned A20 Ironclad public observations, backward map DAG reasoning, and pooled/attention card–relic variants.
- `learn.py`: learner; executable entrypoint remains `apps/run_rl/train.py`.
- `architectures/`: named architecture specifications and their freezing rules.

Old model kinds, parameter names and checkpoint layouts are retained. v3 uses new kinds and requires newly collected graph observations. It supports Acts 1–3/Heart inputs, but is untrained: implementation and plumbing tests are not evidence of full-game strength.
