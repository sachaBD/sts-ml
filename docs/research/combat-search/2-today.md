# 2. Legacy guided-rollout search (reference only)

This describes the teacher, not the new dedicated [PV search](../../../agents/combat/pv/README.md).

Guided-rollout MCTS (`PublicBeliefCombatSearch` in sts_lightspeed; settings in
`agents/combat/search/teacher_leaves.hpp`). Terms: [1-tree-search-basics.md](1-tree-search-basics.md).

- **Selection:** UCB1, no prior. A move never tried is always tried first, so every node spends its first
  visits on every legal move once.
- **Evaluation:** a rollout by a handwritten policy to the end of the fight.
- **Randomness:** 8 worlds sampled from what the player can see; nodes keyed by what the player sees.
- **Budget:** 500 / 5k / 10k / 10k / 20k simulations (easy / hard / elite / event / boss). A fresh tree for every decision.
- **Known:** 100k simulations were no better than 20k on Act 2 bosses
  ([act2-boss-search](../../../experiments/act2-boss-search/REPORT.md)). Estimated (not measured) depth of real
  search: ~1–2 turns; the rollout policy decides the rest of the fight.
