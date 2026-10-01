# Act 1 card selection

Planning notes. Nothing here is implemented yet.

## Goal

Ultimately: maximize Ironclad A20 Heart win probability.

First bounded problem: improve ordinary **card-reward choices** (take one offered card or skip) during Act 1, with a **frozen combat agent** and **fixed controllers for every other decision** (map, rest, shop, events: `SimpleAgent` via `environments/overworld/act1_run.cpp`).

Act 1 boss survival is a provisional development objective, not proof of better full-game play: it can favour decks that fail later.

## Where we start

- **Baseline selector:** `SimpleAgent::stepCardReward` (`../sts_lightspeed/src/sim/search/SimpleAgent.cpp`): a static per-card priority table (`cardPriorityMap`) plus a max-copies cap. Upgraded cards are slightly preferred. It is blind to deck synergy, HP, relics and boss.
- **Combat:** the value-net-guided teacher search, ~2 s per fight at 2k simulations ([combat results](../act-1-combat/2026-09-26-mirror-mcts/REPORT.md)).
- **Competitor:** `../sts_ml`, a separate and currently stronger full-game agent. It is the bar to beat, not a component. Its card picker is summarized in [approaches/human-imitation](approaches/human-imitation/README.md).

## Approaches

Each approach has its own directory. Status reflects the current discussion, not results.

| Approach | One line | Status |
|---|---|---|
| [CARDS](approaches/cards/README.md) | Combat-Approximated Rollouts for Deck Selection: exact simulator + learned combat-outcome model, rollouts per choice, distil into a picker, iterate | **Main direction** |
| [Gauntlet](approaches/gauntlet/README.md) | Score each candidate deck on a weighted set of likely fights (real or modelled combat); pick the best | **Option / first step**: cheap, doubles as the per-card signal test |
| [Run value](approaches/run-value/README.md) | Learn V(post-choice run state) from real run outcomes (TD-Gammon style); pick argmax | Alternative; also CARDS' eventual cutoff value |
| [Model-free RL](approaches/model-free-rl/README.md) | PPO-style RL on run-level decisions with real combat | Not planned: sample-inefficient on one machine |
| [Human imitation](approaches/human-imitation/README.md) | Imitate human Heart-winner choices (the competitor's picker) | Reference only; no data, not win-grounded |
| [LLM picker](approaches/llm-picker/README.md) | Prompt an LLM with the run state and offer | Option: prior / sanity check |

The approaches compose rather than compete:

- **Real-combat continuations** are the unbiased but expensive reference that every approach is audited against.
- **Gauntlet** is CARDS with no future (no later picks, no map, no HP carried between fights).
- **Run value** is what CARDS needs at its rollout cutoff once the horizon extends past Act 1.

## Key risk shared by the model-based approaches

One card changes a fight's outcome by a few HP on average. A combat model's absolute error on a new deck is probably larger. What matters is its error on the **paired difference** (same fight, deck with vs without the card), where shared error can cancel. This is unmeasured. The first experiment should measure the real per-card effect before any model is built (see [Gauntlet](approaches/gauntlet/README.md)).

## Common evaluation rules

- **Primary evidence:** actual played-through outcomes. Surrogate scores and imitation accuracy are diagnostics only.
- **Pairing:** pair alternatives by reward state and full-run candidates by seed. Split by source run or seed *before* branching, and keep untouched confirmation seeds.
- **Compute:** fix per-method compute budgets explicitly.
- **Information:** use only public information. Simulator cloning is an experimental tool, not permission to see hidden futures (the RNG seed is unknown to the agent; fights are stochastic).
- **Success:** a new selector improves held-out real Act 1 outcomes under identical downstream controllers at an explicit compute budget. Check longer-horizon outcomes before heavily optimizing the Act 1 endpoint.
