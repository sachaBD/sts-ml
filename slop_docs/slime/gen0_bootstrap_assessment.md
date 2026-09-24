# Gen 0 value-search assessment

## Conclusion

The current neural agent significantly underperforms the guided-rollout teacher, but this experiment does **not** isolate network quality. It simultaneously replaces rollout evaluation and reduces simulations per decision by 50×.

The strongest suspects are:

1. Too little search in the evaluated neural configuration.
2. Insufficient training coverage of positions visited by neural search.
3. Prediction errors that matter more for move ranking than average validation loss suggests.

There is one confirmed encoding omission, but no demonstrated catastrophic inference-plumbing bug. A larger network is not the first recommended change.

## Recorded experiment

Sources:

- `runs/gen0-value-v1/value_checkpoint.json`
- `runs/gen0-value-v1/training.log`
- `runs/gen0-eval-v1/episodes.jsonl`

| Agent | Simulations per decision | Wins |
|---|---:|---:|
| Guided-rollout MCTS | 20,000 | 45/58 |
| Neural-leaf MCTS | 400 | 27/58 |

These are 29 held-out decks with two evaluation seeds each. Neural search lost 18 fights that the baseline won and rescued none. Seeds within a deck should not be treated as fully independent when estimating uncertainty.

The result establishes that this particular configuration is weaker. It does not establish that neural evaluation is weaker at equal simulations or equal wall-clock time.

### Small diagnostic budget comparison

Four completed matched diagnostic fights gave:

| Configuration | Wins |
|---|---:|
| Rollout, 20,000 simulations | 3/4 |
| Rollout, 400 simulations | 1/4 |
| Neural, 400 simulations, batch 64 | 2/4 |
| Neural, 400 simulations, batch 1 | 2/4 |
| Neural, 2,000 simulations, batch 64 | 4/4 |

This is a tiny, non-random diagnostic subset, not a performance estimate. It motivates a controlled budget sweep, not a claim that 2,000 simulations solves the problem.

Diagnostic binaries and scripts were created under `/tmp/sts-bootstrap-audit/`; production implementation files were not changed. Diagnostic binaries imposed a 250-decision cap, so unfinished capped fights must not be confused with actual combat losses. The four completed comparison triples above did not hit that cap. A separate trace binary used a 100-decision cap.

## Training data and generalization

The complete dataset contains 386 fights across 193 decks. Training uses 328 fights across 164 decks, yielding 11,041 rows. Validation contains 2,213 rows from 29 decks.

| Prediction metric | Training | Validation |
|---|---:|---:|
| Mean absolute error | 0.03503 | 0.08365 |
| Mean squared error | 0.002177 | 0.011945 |

The model fits training data substantially better than unseen decks. This points toward generalization and coverage problems, rather than clear evidence that the architecture cannot fit the task.

Rows from one fight are correlated; 11,041 rows do not represent that many independent combat examples.

On validation rows with multiple actions, the median absolute difference between the teacher Q values of the two most-visited moves was about 0.00365. Absolute state-value error is not the same as action-ranking error—errors can cancel across related positions—but this shows why low-looking regression losses can still be inadequate for gameplay.

### Distribution mismatch

Training records positions encountered while the strong teacher plays, with at most one random move per fight. Neural search evaluates many alternative action sequences, including bad or unusual positions not represented by those trajectories.

Search can exploit prediction errors: it finds a position the network overvalues, then repeatedly steers toward it.

Evidence of problematic behavior:

- One neural evaluation fight required 985 decisions and 573 seconds before winning.
- A diagnostic trace showed prolonged defensive play with Barricade/Entrench.
- Training positions reached at most turn 30; the diagnostic trajectory reached turn 93.
- Maximum player block in training rows was approximately 73; the diagnostic trace exceeded that range.

This does not prove a single causal explanation, but it directly demonstrates out-of-training-range play and very long fights.

**More data is warranted, but prioritize alternative/search-leaf positions, failures, and long defensive trajectories—not just additional rows from successful teacher play.**

## Correctness audit

### Confirmed encoding omission

Changing the three Discovery offers while keeping the rest of the position fixed produces identical serialized NN state inputs.

- Offered cards are represented in action tokens.
- Those action tokens are not serialized into the state consumed by the value model.
- The model also only pools the four ordinary card zones, not the offered zone.

Therefore the value network cannot distinguish those different choice screens. Fixing only the pooling layer would not repair the missing serialized input.

There were no Discovery-selection rows in this dataset, so this is not a demonstrated cause of the aggregate performance gap.

Relevant files:

- `combat/environment.cpp`
- `combat/encoding.hpp`
- `python/sts_combat_rl/models/deep_sets.py`

### Checks that passed

- Batched versus individual predictions differed by at most approximately 1.6e-7 on the checked examples.
- Inspection found matching leaf-request and response ordering.
- Terminal leaves use the search's terminal evaluator.
- All four existing C++ tests passed.
- No obvious sign reversal or accidental use of legacy −1-loss targets was found in this experiment.

These are bounded checks, not proof that every state or search path is correct. The existing tests do not comprehensively validate the new neural-PBCS integration.

## Other concerns

### Large batches with a small search budget

The implementation requests 64 leaf evaluations at a time, with only 400 simulations per decision. Pending evaluations affect selection before their values arrive. This changes search behavior relative to sequential feedback.

Batch-size ablation is justified. Batch 1 did not improve the aggregate result on the four diagnostic fights, and it was slower; batching is not established as the main cause.

### Unvalidated target blend

The trainer uses a 50/50 blend of terminal outcome and root estimate after the random move, and root estimates alone at or before that move.

This is a reasonable experimental target, not a validated optimum. Search estimates include exploration and rollout-policy effects. Terminal outcomes describe the actual continuation. The teacher also uses a small turn penalty and loss shaping absent from the stored terminal label.

Compare pure root targets against the blend, and align objective definitions before introducing more complicated mixtures.

Relevant files:

- `python/sts_combat_rl/training/data.py`
- `python/sts_combat_rl/training/train_value.py`
- `apps/generate_entry_mcts_records.cpp`
- `apps/play_entry_pbcs.cpp`
- `python/sts_combat_rl/training/evaluate_entry_pbcs.py`

## Recommended next experiments

1. Keep the existing model initially. Sweep rollout and neural simulation budgets on matched decks/seeds; report both simulation counts and wall-clock cost.
2. Test short guided rollouts followed by neural evaluation; see the companion hybrid-search note.
3. Collect strong-teacher labels for sampled alternative positions that neural search actually visits. Keep held-out decks out of training.
4. Compare pure root labels with the current blend.
5. Fix missing choice-screen information and add regression tests.
6. Add explicit time/decision limits and report unfinished fights separately, without silently dropping them or claiming they are ordinary losses.

Consider architecture changes after these controls. Current evidence is not sufficient to conclude that sum pooling or width 64 is the principal bottleneck.
