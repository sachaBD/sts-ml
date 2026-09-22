# Slime Boss roadmap

Current state: a value net inside the teacher search (`PublicBeliefCombatSearch`, native C++ inference) at 2k simulations roughly matches the 20k rollout teacher on held-out decks, at about 2 s per fight.

## Next

1. **Gen 1 data:** generate with the net-guided search as teacher, with early stopping and child rows. Retrain on the combined data. Evaluate on the fixed held-out decks.
2. **Deeper training positions:** store well-visited positions two or three levels down the search tree, not only root children, to cover what search actually queries.
3. **Turn penalty in the label:** stop the net preferring slow wins, which block-stacking decks exploit (the 50-turn cap is only a guardrail).
4. **Tree reuse:** keep the played move's subtree between decisions (`rebase`).
5. **Full entry states:** encode relics and potions, and drop the `deck_hp_only` projection.

## Later

- Policy head trained on root visit counts (`actions` column), used as a PUCT prior.
- Pull-queue workers instead of static root splits.
- Other encounters beyond Slime Boss.

## Evaluation

Every change is judged on the same held-out decks (split by `deck_signature`), paired against saved teacher results. Report wins, paired win/loss swaps, final HP difference and time per fight.
