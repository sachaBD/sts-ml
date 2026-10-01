# Next step after the value-input fix: policy-guided elite search

**Wait until the value-net state representation work is finished.** This is a proposed next experiment, not permission to start it now.

## Idea in plain English

Keep MCTS choosing moves. Give it an initial hint about **which legal moves deserve attention**. The hint is a *policy*, not a command: search must still explore alternatives and may override it.

At the **root** (the real decision in front of the player), a policy might favour Defend over Strike. MCTS still checks both and chooses its real move after searching. Our existing search has an optional root-prior mechanism, but value play does not currently provide learned priors. That mechanism guides only the root; guiding imagined decisions deeper in the tree would be a later step.

## Proposed model

Add a **policy head to the improved value net**, sharing its state encoder:

- Value output: estimated eventual fight score from this state.
- Policy output: one preference score **per currently legal action**. Include card plays with their targets, potions, end turn, and card-selection choices. Do not use a fixed output slot per card name.

Initialize the shared encoder/value components from the corrected value checkpoint if compatible. If training the policy and value together harms value quality, consider separate models later; start with one shared model.

## Training signal

Use strong MCTS decisions: target the *distribution of visits across legal root actions*, rather than just the single move MCTS finally chose. Keep the legal-action mapping identical between recorded targets and inference. Prefer data produced with a search budget relevant to the 20k evaluation; quantify the available 20k data before committing to new collection. The value head should continue to predict the same terminal fight score used by search (death 0; wins reward remaining HP and potions). Do not claim that a lower policy loss proves better play.

## Low-risk rollout

1. Wire learned policy preferences into the **existing root prior** only. Keep exploration and allow search to override the prior; compare identical states with prior strength zero vs positive.
2. Test on the **frozen bucket-5 elite validation fights** at 20k sims, comparing policy+value, value-only, and guided-rollout MCTS. Report paired HP-equivalent, win/death rates and Nob/Lagavulin/Sentries separately. Do not use the reserved bucket-2/3 final fights for tuning.
3. If root guidance helps, extend it to **nodes deeper in the tree**. Root-only guidance is an intentionally limited pilot, not full AlphaZero-style policy-guided search.

The important distinction: our value-only search **already** ranks explored branches by backed-up value plus an exploration bonus. The policy head adds a *prior before a branch has enough search evidence*. Taking the max of immediate value predictions instead is not the same thing and can amplify value errors.

## Pointers

- Simple search glossary: `slop_docs/combat_search_in_plain_english.md`.
- Value-input changes are a prerequisite: `combat/environment.cpp` (`encode_state`) and the value-model encoder. Coordinate with the owner/NN agent before touching their files.
- Search's optional root prior: `sts_lightspeed/include/sim/search/PublicBeliefCombatSearch.h` (`setRootPrior`); root selection: `sts_lightspeed/src/sim/search/PublicBeliefCombatSearch.cpp` (`select`). Current default `priorStrength = 0`.
- Existing experiment and reserved evaluation lists: `experiments/act-1-combat/2026-09-28-elite-specialist/README.md`.

**Unresolved before implementation:** how to represent and record variable legal actions for training and inference; how many independent 20k teacher fights are available; policy-prior strength and its effect on value calibration. Decide these explicitly rather than silently using action names or only chosen moves.
