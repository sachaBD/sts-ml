# Overworld value agent

Owns the `run_policy_v1`/`run_policy_v2` architectures, model-input encoding, after-state policy, TD(lambda) targets, checkpoint loading, and learning.

- `core.py`: model construction, batching, targets, and checkpoint/log helpers.
- `policy.py`: chooses cards and other overworld decisions, including sampled-lookahead evaluation.
- `learn.py`: learner; executable entrypoint remains `apps/run_rl/train.py`.
- `architectures/`: named architecture specifications and their freezing rules.

Model kinds, parameter names, and checkpoint dictionaries are unchanged by the directory reorganization. The agent is trained on Act 1 outcomes; that is not evidence of full-game strength.
