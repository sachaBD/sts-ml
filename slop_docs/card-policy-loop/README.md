> **Superseded (2026-10-01):** the active approach is `slop_docs/run-rl/README.md` (run-level RL on real games). Kept for the record.

# Card-selection agent: four models and one training loop

*Planning document, 2026-09-30. Status: design agreed at a high level; nothing below the "Built so far" line exists yet.*

## The goal in one paragraph
We want an agent that picks cards well in Slay the Spire (Ironclad, Ascension 20, Act 1 first). A card pick is hard to
judge: its value only shows up over the many fights that follow, and each of those fights is random and expensive to play
out with our search-based combat agent. The plan is to judge picks by **simulating the rest of the run cheaply**. A
learned model stands in for expensive fights. A learned policy makes the future picks inside those simulations. Then the
policy learns from the results, and the loop repeats. This is the *expert iteration* pattern (as in AlphaZero): search
produces better decisions than the policy alone, the policy is trained to imitate the search, and a better policy makes
the next search better.

---

## The four agents

| # | Agent | Question it answers | Input → output | Cost per call | Role |
|---|---|---|---|---|---|
| 1 | **Combat player** | "How do I play this fight?" | full fight state → actions (tree search, MCTS) | seconds (boss ≈ 6 s) | **Ground truth.** Plays real fights. Everything else is checked against it. |
| 2 | **Combat outcome model** | "If I enter this fight with this deck, what happens?" | deck, relics, potions, HP, encounter → P(win), distribution of HP after the fight | microseconds | **Stand-in for fights** inside simulations. |
| 3 | **Card policy** (likely the same network as 4; see deep dives) | "Which of these 3 cards (or skip) should I take?" | run state + offered cards → a probability for each option | microseconds | **Makes picks** inside simulations. This is the product we are building. |
| 4 | **Run value model** | "From here, how likely am I to clear the act?" | run state (deck, HP, floor, gold, known boss, …) → P(act clear), expected HP at the boss | microseconds | **Ends simulations early**, so we don't play every simulated run to the end of the act. |

Notes:
- Agents 3 and 4 can share one network: one encoder with two output heads, a policy head and a value head.
- Agents 2, 3 and 4 all read a deck. They should share the card/deck encoder design that worked best in the outcome-model
  sweep: each card is encoded together with the fight it is in, then all cards are pooled (sum + mean).
- Everything outside combat and card picks (map path, rest vs upgrade, shops, events) starts as **fixed hand-written
  rules**. Those decisions can join the policy later.

## The training loop

```
            ┌──────────────────────────────────────────────────────────────────────────┐
            │                                                                          │
            ▼                                                                          │
 (A) SEARCH: at each card pick, score every option by simulated rollouts               │
     - fights resolved by the Combat outcome model (2)                                 │
     - future picks made by the current Card policy (3)                                │
     - rollouts stopped at a horizon and scored by the Run value model (4)             │
            │                                                                          │
            ▼                                                                          │
 (B) LEARN: train Card policy (3) toward the search's option values;                   │
     train Run value (4) toward observed outcomes                                      │
            │                                                                          │
            ▼                                                                          │
 (C) REAL PLAY: play full runs with the new policy and the real Combat player (1)      │
     - the gate: new policy vs previous policy, same seeds, real act-clear rate        │
     - produces real fights with the decks the new policy actually builds              │
            │                                                                          │
            ▼                                                                          │
 (D) CORRECT THE SIMULATOR: retrain Combat outcome model (2) on the new real fights ───┘
```

### (A) Search: how one card pick is evaluated
1. **Options:** the 3 offered cards plus skip.
2. **Rollouts:** for each option, play the rest of the act forward many times inside the *macro simulator*. The game
   simulator handles everything except combat. Fights are sampled from the outcome model: draw win or loss and the final
   HP from its predicted distribution, not its average. Later picks are made by the current policy, and other decisions by
   the fixed rules.
3. **Common random numbers:** every option is simulated against the *same* futures: same map, same encounters, same
   future card offers, same fight-outcome random draws. Then the only difference between options is the pick itself.
   This matters most for making the search work: a single pick moves the act-clear rate by only a few percent, and
   without shared futures that signal drowns in noise.
4. **Truncation:** stop each rollout after a horizon (e.g. a few floors or the next elite) and score the rest with the run
   value model (4). This lowers variance and cost.
5. **Model uncertainty:** in each rollout, use a randomly chosen member of the outcome-model ensemble (several training
   seeds). The search then cannot lean on a single model's mistakes.
6. **Result:** an estimated value (Q) for each option. The move actually played is sampled from a softmax over the Q
   values, with some noise so the policy keeps exploring. The full Q vector, not just the choice, is saved as a training
   target.

### (B) Learn
- **Card policy (3):** trained to match the search's preferences, i.e. the cross-entropy between the policy and a softmax
  over the search's Q values. Learning from all options' values gives far more signal per search than learning only
  which option won.
- **Run value (4):** trained on what actually happened afterwards, in real runs and in rollouts. Real outcomes should
  carry more weight.

### (C) Real play, the gate
- Full Act 1 runs with the **real combat player**: new policy against the previous policy on the same game seeds.
- The **headline metric** is the real act-clear rate. Secondary metrics are HP at the boss and HP at the end of the act.
- Only a policy that wins this comparison moves on. A real Act 1 run is about 15 fights, roughly 20 worker-seconds, so a
  few thousand real runs per iteration is affordable.

### (D) Correct the simulator
- The new policy builds decks the outcome model may never have seen. The model will be wrong there, and the search will
  exploit those errors if they are not fixed.
- Retrain the outcome model on the real fights from (C). Add targeted real fights where the policy spends time and where
  the ensemble members disagree most.

## Deep dives

### Stopping early at the value estimate (truncated rollouts)
- **Problem:** a rollout played to the end of the act (about 10 fights, 6 picks) ends in one coin flip, cleared or not. A
  single pick shifts that coin by maybe 1–3%, so separating options needs thousands of full rollouts per option.
- **Idea:** play only `h` floors, then score the state with the run value model:
  `Q(A) = mean over rollouts of [0 if dead before the horizon, else V(state at horizon)]`.
  - V is a probability (e.g. 0.63), not a 0/1 outcome, so the randomness after the horizon is averaged inside V instead of
    sampled. Illustrative: the outcome's standard deviation falls from about 0.5 to about 0.15, i.e. roughly 10× fewer
    rollouts, and each rollout is shorter.
- **Trade-off: bias vs noise.**
  - `h = 0`: `Q(A) = V(deck + A)`. Cheapest; any error in V goes straight into Q.
  - `h` = a few floors, or up to the next elite or boss: the usual sweet spot.
  - `h` = end of act: unbiased, noisy and expensive.
  - Choose `h` by measurement: the shortest horizon whose option ranking agrees with very long rollouts on a sample of
    decisions.
- **Where V comes from:**
  1. **v0, no training:** chain the outcome model along the likely route, e.g. P(clear) ≈ Π P(win upcoming
     elites/boss), carrying HP forward from the predicted HP distributions.
  2. **Learned V:** trained on what actually followed a state, in rollouts and in real games.
  3. **Retrain every iteration:** V means "value under the current policy", so it goes stale when the policy improves.
- **What V must see:** deck, relics, potions, HP and max HP, gold, floor, the known act boss, and the map ahead (elites,
  rests and shops on the path).

### Training on each option's value (not just the winner)
A search yields, per decision: the state, the options, a value estimate (Q) for each option, and its standard error.
With common random numbers, the error on differences between options is small even when the levels are noisy.

| Target | Learns | Problem |
|---|---|---|
| Winner only (one-hot) | "A was best" | Discards margins; noisy winners become hard labels. |
| Soft target, softmax(Q/τ) | a ranking with margins | τ must be tied to the noise; absolute values are lost. |
| **Regress Q(s,a) for every option** (recommended) | how good each option is | Needs scale and noise handling (below). |

- **Card picks are deterministic steps:** taking A just gives deck + A, so `Q(s, A) = V(state after taking A)`. The card
  policy (3) and run value (4) can therefore be **one network**. The policy is softmax/argmax of V over the after-pick
  states. Training targets: search values for the after-pick states, and real/rollout outcomes for all states. A separate
  fast policy head is optional later.
- **Train on differences between options:** the loss compares the network's predicted gap between two options
  (`V(s+A) - V(s+skip)`) with the search's measured gap. Absolute levels are learned from real outcomes.
- **Weight by confidence:** weight each pair by 1/SE² of its measured difference; decisions where all options are within
  noise barely count.
- **Explore when acting, not when learning:** noise decides which option gets played; the training target is always the
  full vector of option values.

## Why this should work, and where it can fail
| Risk | Symptom | Mitigation |
|---|---|---|
| **Noise:** pick effects are small, run outcomes are coin flips | search picks differ between repeats of the same decision | common random numbers, truncation by the value model, train on full Q vectors, measure repeat-agreement |
| **Simulator exploitation:** the policy favours decks the outcome model overrates | simulated clear rate rises, real clear rate doesn't | real-play gate every iteration, ensemble sampling, targeted real fights, track the simulated-vs-real gap |
| **Missing fight consequences:** the outcome model predicts win and HP only | potions, max-HP gains, relic counters wrong in rollouts | predict the post-fight state (HP, then potions, then the rest) or approximate it by rule |
| **Fixed rules for non-card decisions** limit the ceiling | good picks, poor paths or rests | bring those decisions into the policy after card picks work |
| **Circularity:** the policy, value and simulator all learn from each other | quality drifts without anyone noticing | real play with the real combat player is the only accepted evidence of progress |

## Build order
0. **Policy v0, no training:** score each option with the outcome model directly, i.e. expected outcome over the fights
   left in the act, weighted toward the known boss. It is a myopic baseline and the first policy used inside searches.
1. **Macro simulator:** game simulator for everything except fights, fights sampled from the outcome model, fixed rules
   for non-card decisions, common-random-number control over all randomness.
2. **Search:** per-pick rollouts with the v0 policy; check that repeated searches on the same decision agree.
3. **Policy/value network (3+4)** trained from the search; first real-play gate against v0 and against the current picker.
4. **Close the loop:** real fights from (C) retrain the outcome model (2); repeat.
5. **Widen:** more decisions (path, rest, shop), then Act 2+.

**Success criterion for the first cycle:** the trained policy beats both v0 and the existing card picker on real Act 1
A20 clear rate, over enough paired runs that the gap is outside the noise.

---

## Built so far (as of 2026-09-30)
- **Combat player (1):** search-based teacher, used to generate all fight data.
- **Combat outcome model (2):** the recommended model is `co2-w32-h64-l1-d30` (`topology/combat_outcome/`), trained on
  about 330k synthetic fights plus natural fights.
  - On a held-out set that plays each (deck, fight) 16 times:
    - Elite win-rate error is about 13 points; the model captures about 70% of the real differences between states.
    - Boss win-rate error is about 27 points (about 44%).
    - Using the model to choose among 4 candidate cards + skip captures about 90% (elite) and about 65% (boss) of the
      value available relative to a noisy oracle.
  - Caveat: these numbers are somewhat optimistic, because the model was chosen on the same data.
  - See `experiments/card-outcomes/REPORT.md`, `experiments/topology-sweep/REPORT.md`, and
    `experiments/topology-sweep/diag/objective-hs16.md`.
- **Known limits of (2):**
  - Boss estimates are coarse.
  - It predicts no post-fight state beyond HP.
  - The training data comes from few distinct starting decks. That, not model size, is the current bottleneck.
- **Win-rate UI** for (2): `gui/` (see `experiments/outcome-ui/NOTES.md`).
- **Card policy (3) + Run value (4):** topology only, untrained: kind `run_policy_v1`
  (`python/sts_combat_rl/topology/run_policy_v1.py`), spec `topology/run_policy/rp1-w32-h64-l1-d30.toml`,
  Every option is scored as an after-state (deck + card; skip = unchanged), with a
  value head (P(act clear), HP bins) and a policy head (probabilities over cards + skip; sample / greedy / value-greedy).
  Map = the enumerated remaining paths (current node -> boss, deduplicated by room sequence). Each path is embedded
  (rooms + floors), scored for this deck/HP after-state, and max-pooled as a whole path (the best route's embedding + its
  score + the mean path). The route score can later be supervised and doubles as a path-choice head. Reasoning:
  `map-encoding.md`.
- **Macro simulator + search:** built; first results in `search-first-results.md` (searches agree 99% at 500 rollouts; common random numbers 13× variance reduction; simulator optimistic: SimpleAgent 71.5% sim vs 60.6% real).
- **Trainer, real-game gate:** not started.

## Open questions
- Answered 2026-09-30, see `macro-simulator-scoping.md`:
  - sts_lightspeed runs full runs outside combat, with a skip-battle hook and copyable state.
  - There are no Python bindings.
  - Common random numbers need our own per-floor reseeding.
  - Baseline: SimpleAgent card picks + MCTS combat = **60.6% ± 1.5 Act 1 A20 clear rate**.
- Is Act 1 clear rate (plus HP at the boss) the right headline metric until the loop is proven?
- Where does the search loop live: C++ with an exported outcome model, or Python with batched model calls?
- Earlier related notes: `slop_docs/card-selection/`.
