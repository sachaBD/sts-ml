# Agents

Agents own what to choose and how that chooser is learned: model architecture, inference, model-specific batching, targets/losses, learning, and weight export.

- `combat/search/`: teacher search and leaf evaluation; depends on the combat environment and value model.
- `combat/value/`: Deep Sets value/policy models, native inference, learning, and export. Model kinds and native weight formats remain versioned and frozen once used.
- `overworld/value/`: learned after-state chooser, run-policy architectures, feature preparation, TD targets, checkpoints, and learning.
- `overworld/search/`: model-based paired card-choice rollouts and their rollout policy; uses the shared combat-outcome predictor.
- `overworld/neow.py`: experimental Neow bandit policy (retained for earlier experiments).

Apps own CLI/config handling, worker execution, collection workflows, and output orchestration. Saved weights and histories belong in `runs/`, not here. Tests live alongside their owners.
