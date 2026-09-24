# Hybrid rollout and neural-value search

## Yes: this is an established approach

There are two related designs worth distinguishing.

### 1. Truncated rollout, then a value estimate

```text
MCTS follows a path through its tree
                ↓
          reaches a new leaf
                ↓
  play a short guided rollout in the simulator
                ↓
       has the fight finished?
          /             \
        yes              no
         ↓                ↓
  exact final score   V(resulting state)
          \              /
           back up the score
```

A rollout does not have to continue to the end. It can stop at a fixed depth or other boundary and use an approximate value function for the remaining future.

This is commonly described as **truncated rollout with terminal value approximation**, or **bootstrapped rollout evaluation**. It belongs to the broader rollout/approximate dynamic programming literature; see Bertsekas and Tsitsiklis, *Neuro-Dynamic Programming* (1996), and Bertsekas, *Rollout, Policy Iteration, and Distributed Reinforcement Learning* (2020).

For a general reward process, the backed-up estimate is:

```text
rewards collected during the short rollout
    + discounted value of the stopping state
```

For our terminal-score combat objective, an unfinished rollout can simply return the stopping state's predicted final score, provided the network and search use consistent scoring. Do not add current HP as another reward: the predicted terminal score already accounts for surviving HP.

### Why it could help here

- The simulator resolves immediate damage, enemy turns and other near-term consequences instead of asking the network to estimate them immediately.
- Guided actions can move a strange search leaf toward positions more like those seen during training.
- Terminal outcomes reached during the short rollout remain exact.

### What it does not guarantee

- A heuristic rollout can miss strong continuations or introduce its own bias.
- The stopping positions can still differ from the training distribution.
- More simulation work may reduce how many tree simulations fit in a time budget.
- It does not replace the need for good value labels and adequate coverage.

## 2. Mix a full rollout result with a neural estimate

The original **AlphaGo** used a related but different hybrid: it combined a value-network estimate at a leaf with the result of a fast rollout to the end of the game.

```text
                       leaf
                      /    \
                 V(leaf)   full rollout → outcome
                      \    /
                    weighted mix
                         ↓
                   back up the score
```

Reference: Silver et al., *Mastering the game of Go with deep neural networks and tree search*, Nature 529, 484–489 (2016), DOI: https://doi.org/10.1038/nature16961.

This is not the same as “roll out a little, then call the network”: AlphaGo's hybrid combined two estimates. Later AlphaGo Zero removed rollout evaluation and relied on the learned network with search.

A training-label blend is also a separate choice: mixing labels during training does not itself make the deployed search a rollout/value hybrid.

## Our code already has the mechanism for truncated rollouts

In `apps/play_entry_pbcs.cpp`, neural search currently requests:

```cpp
search.requestBatch(batch, simulations, 0, 0);
```

The final two arguments bound rollout turns and rollout actions. Both are zero, so the guided rollout stops immediately and the network evaluates the newly expanded leaf.

`PublicBeliefCombatSearch::simulate` and `boundedRollout` in the shared simulator already support doing guided moves before issuing the neural request. If the rollout finishes the fight, the search backs up its terminal score without requesting a neural value.

An example experimental setting is:

```cpp
search.requestBatch(batch, simulations, 1, 16);
```

This allows at most one turn increment and 16 rollout actions before requesting a value. These are illustrative settings, not tuned recommendations. Check the stopping state's semantics: a turn-increment limit is not necessarily equivalent to “one complete future player turn”. A zero turn limit stops immediately even if the action limit is positive.

## Suggested controlled test

Use the same checkpoint and held-out deck/seed pairs throughout:

| Arm | Leaf evaluation |
|---|---|
| A | Immediate network evaluation: current implementation |
| B | Guided rollout until one turn increment, capped at 16 actions, then network |
| C | Longer bounded guided rollout, then network |
| D | Full guided rollout: existing teacher baseline |

Measure:

- Wins and surviving HP on paired fights.
- Wall-clock time and simulations per decision.
- Neural calls versus exact terminal evaluations.
- Decision/turn counts and unfinished fights.

Compare at equal simulations first to isolate evaluator behavior, then at comparable wall-clock budgets for practical utility. Keep this ablation separate from model retraining so its effects remain interpretable.

**Recommendation:** try bounded guided rollouts before redesigning the network. It is a small integration change with a well-established conceptual basis, but its benefit here still needs measurement.
