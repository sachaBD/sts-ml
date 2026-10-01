# Combat search, in plain English

## Units of play and search

1. **Fight:** From combat start until the player dies or all enemies are defeated.
2. **Turn:** Player moves until they end the turn; then enemies act. A turn may contain several decisions.
3. **Decision:** One player choice, such as playing a card, using a potion, ending the turn, or making a required card selection. The agent searches again at each real decision.
4. **Search tree:** Imagined futures starting at the current real decision. An action is a branch; a resulting fight state is a node. These are *not* moves made in the real fight.
5. **Leaf:** The state where one trip through the search tree stops. The search needs a score for it.
6. **Rollout:** Further *imagined* play after a leaf, choosing moves using the handwritten policy. This may involve many decisions and turns; its extra moves are not one tree simulation apiece.
7. **One simulation (sim):** Start at the current decision, select imagined actions down the tree to a leaf, get a score, and pass that score back along the selected branches. It does **not** immediately make a real move.
8. **Search budget:** The maximum number of sims for *one real decision*, not for a whole turn or fight. Search may stop early.

## What changes between the two agents?

Both agents use the same kind of search tree and choose a real action **after** searching. They differ in how a nonterminal leaf gets its score:

- **Guided-rollout MCTS:** continue playing imagined actions with the handwritten policy; score the simulated result.
- **Value-net search:** stop at the leaf and ask the neural net to estimate its eventual score.

**Example:** At a real choice between Strike and Defend, one sim might select imagined Defend, continue through the tree, score the leaf, and credit the branches it took. Another sim might select imagined Strike. After searching, the agent chooses **one** real action. The fight then advances to the next real decision, and search runs again.

**20,000 sims does not mean 20,000 imagined moves or states.** It means up to 20,000 trips through the search tree *per decision*. Guided rollouts can simulate additional moves inside each trip; the immediate value net does not. Equal simulation counts are therefore not equal amounts of simulated future or equal computing time.

## Relation to the standard idea

Replacing a rollout with a learned value estimate is a standard way to use a value network in tree search. It is not automatically stronger than the rollout. The network has to predict the consequences of future play from its encoded inputs. Modern AlphaZero-style search also uses a **learned move-policy prior** to guide which branches to explore; our value-net search does **not** add that policy prior. See `slop_docs/slime/hybrid_rollout_value_search.md` for the separate option of a short rollout followed by a value estimate.
