# Open questions / next decisions

## User decisions

1. Which combat policy should the surrogate represent first? Recommendation: existing fair guided-rollout MCTS with an explicitly documented encounter-dependent budget, using compatible existing data. Do not wait for a new stronger agent.
2. First application: recommend fast evaluation of the existing gauntlet, not full macro rollouts. Confirm whether this matches the intended immediate use.

## Implementation findings to resolve

- Replay extraction: implementation session reports existing replay helpers can recover pre-init GameContext and post-exitBattle persistent state by replaying recorded actions without combat search. Verify supported source types, resampled HP/potions and simulator drift; not every historical record is guaranteed to replay exactly.
- Source policy metadata may be incomplete in interrupted runs. Choose an explicit initial cohort, not all data blindly.
- Inspect the endpoint relative to healing and rewards, and encode both raw battle outcome and post-exit macro HP if cheap. Prefer absolute ending HP to ambiguous 'HP lost'.
- Recover full terminal inventories/deltas during extraction if inexpensive, even if first model learns only survival/HP. This avoids repeating extraction when extending the target.
- Decide output distribution after inspecting HP/max-HP support. No elaborate joint distribution required initially.

## Research questions (not implementation blockers)

- Do paired card effects generalise beyond SimpleAgent-generated decks?
- Does future-reward augmentation improve the gauntlet's decision usefulness?
- When does a cheap gauntlet suffice, versus needing simulated macro continuations?
- Which omitted persistent transitions materially change choices?

## Implementation-session evidence, pending local reproduction

`combat-transition` reports substantial random-move contamination in bootstrap fights, cleaner value_play data, repeated starts across replay runs, and recoverable terminal state via `exitBattle`. These observations motivate filtering and replay extraction. Exact cohort counts belong in the extractor's report, not as fixed dataset facts here.
