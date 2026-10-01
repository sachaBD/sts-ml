# Small problems and how they compose

## Shared vocabulary

- **Run state:** deck (including upgrades), HP/max HP, relics and relevant counters, potions, floor, known boss, and other relevant public context. Not just the deck.
- **Action:** take one offered card or skip; enumerate actual legal choices rather than hard-code four.
- **Continuation policy:** the fixed combat, future card-selection, and other controllers used after the candidate action.
- **Endpoint:** initially proposed as Act 1 victory or death. Reaching the endpoint supplies a binary outcome.

The quantity we want is:

`Q(state, choice) = P(endpoint success | make this choice, then use the continuation policy)`

This is conditional on those controllers and that endpoint. It is not an intrinsic card rating or Heart win probability.

## A. Choice application: exact and cheap

**Problem:** construct each post-choice state correctly.

**Input → output:** pre-choice snapshot + legal choice → post-choice snapshot.

Use game rules, not learning. Preserve all state unrelated to the choice.

**Success:** branch/replay tests verify every legal choice, including skip, without cross-branch mutation or hidden-information leakage.

## B. Real continuation: expensive ground truth

**Problem:** measure what happens after a choice.

**Input → output:** post-choice state + sampled future + fixed controllers → endpoint outcome and trajectory.

Use the actual simulator and frozen combat agent. Non-card decisions still happen, but fixed controllers make them. Future card choices initially use the inherited selector.

**Success:** reproducible continuations, understood runtime, and auditable outcomes. This is the reference for all cheaper approximations.

## C. Combat surrogate: optional acceleration

**Problem:** replace an expensive combat with a cheap sampled outcome.

**Input → output:** combat-start state + encounter + combat-controller version/budget → distribution of combat-end outcomes.

Predict death and relevant survivor outcomes, not just expected HP loss. HP and potion outcomes can be dependent; permanent combat effects must be represented if enabled. Copy unchanged state exactly. Apply rewards and other known game rules outside the model.

The model describes a particular combat agent. Changing that agent can invalidate it.

**Success:** held-out calibration and outcome accuracy, useful speed-up, and—most importantly—good card-choice rankings when checked with B. Historical prediction accuracy alone is insufficient.

**Main risk:** the selector discovers decks on which this model is overoptimistic. Collect real combat outcomes for alternative card additions, not only decks selected by the old policy.

## D. Root-choice evaluator: compare, do not solve the whole tree

**Problem:** estimate which current choice is best.

**Input → output:** reward state + choices + continuation policy → per-choice success estimates and uncertainty.

Run multiple continuations per choice, using B or validated C. Future card choices follow the fixed continuation selector. This is rollout-based policy improvement; full MCTS is optional later.

Use matched sampled futures where valid to reduce comparison noise, while preserving each branch's correct marginal distribution. Do not expose actual hidden seed futures to the selector.

**Success:** affordable comparisons that agree sufficiently with real continuation evidence. Preserve uncertain/tied comparisons rather than forcing confident labels.

## E. Learned selector: amortize the comparisons

**Problem:** choose quickly without repeating expensive rollouts at every reward.

**Input → output:** public run state + offered choices → scores/preferences over legal choices.

Initialize from the inherited selector if practical. Train from D's comparisons; the existing combat value scalar is not a strategic value label.

**Success:** better real held-out continuations, then better actual Act 1 runs. Classification agreement with rollout labels is only a diagnostic.

## Composition

```text
reward state → A: candidate branches → D: compare continuations
                                      ├─ B: real combat (reference)
                                      └─ C: sampled combat outcomes (optional shortcut)
                         comparison data → E: improved selector
```

Everything except combat inside a continuation remains exact game logic plus fixed controllers. Freeze the old selector for label generation; evaluate the new selector separately. If successful, promote it as the next continuation policy and repeat.

Two valid paths:

- **Reference path:** A + B + D → E. Expensive, fewer labels, no learned combat-model bias.
- **Accelerated path:** A + validated C + D → E, continually audited with B.

A separate run-value network is unnecessary if continuations reach the endpoint. Truncating earlier creates another problem: learning a trustworthy continuation value. Do not hide that assumption inside an HP or deck-quality heuristic.
