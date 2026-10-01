# Environments

Environments own state, legal actions, observations, and transitions—not policy choices or learning.

- `combat/`: a single fight's simulator interface, public-state/action encodings, recorded-row schema, and fixed starting-state scenarios.
- `overworld/`: persistent state between fights, legal rest/shop after-states, Act 1 orchestration/replay, and macro simulation with externally supplied fight outcomes.

Real combat and injected/modelled outcomes are alternative ways to resolve fights. The outcome predictor lives in `models/combat_outcome/`; it is not part of the environment.

Tests and checks live beside the code they exercise. This layout does not imply a new unified RL API: existing interfaces and namespaces are retained.
