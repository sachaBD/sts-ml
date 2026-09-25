# Terminal-score-only iteration

Status: speculative next direction, discussed during slime-v7. Not an approved experiment or implementation plan. Finish interpreting the search assessment before choosing this path.

## Motivation

DAgger with scalar teacher-value corrections did not improve gen0. The preliminary 15k → 60k search comparison at eight belief particles also showed no improvement. These results suggest investigating supervision grounded in actual outcomes rather than repeatedly reproducing the rollout teacher's judgments. They do not prove this is the bottleneck; the particle-count and evaluator comparisons remain relevant.

For a single-person, single-machine project, retain the current informative terminal score rather than immediately switching to binary win/loss:

- Loss: 0.
- Win: `(35 + final_hp + 4 * remaining_potions) / (55 + max_hp)`.

Changing the source of supervision and changing the objective are separate experiments. Start by changing only the former.

## Start from gen0, not random

Keep gen0's architecture and weights. The player is MCTS using the value net, not a bare network selecting moves. We already have competent play and useful representations; cold-start learning is unnecessary.

The first candidate is initialized from gen0. Subsequent candidates, if the approach works, start from the current accepted player. Outcome-only supervision does not mean the initialization contains no teacher knowledge.

## Proposed loop

1. **Freeze the current player.** Initially gen0 plus its search settings.
2. **Generate training fights.** Use training-only decks with varied combat randomness. Keep the initial-state distribution stable initially; modest exploration may broaden trajectories without overwhelming the learning signal.
3. **Assign actual terminal scores.** Every visited decision state on a completed trajectory receives that fight's final score. No teacher root-value blend.
4. **Fine-tune a candidate.** Train on the outcome-labelled trajectories, retaining earlier outcome-labelled generations in a replay pool rather than using only the newest batch. The replay mix is a later design choice: old outcomes describe older behavior policies.
5. **Evaluate inside search.** Compare candidate and current player on matched development fights at the same deployment search budget, with exploration off. Judge gameplay, not just regression loss.
6. **Promote selectively and repeat.** Retain the current player if the candidate deteriorates. Do not automatically launch another generation with a worse model. Confirm a selected improvement on fresh held-out fights.

This is approximate policy iteration: evaluate the current search player's outcomes, then use search with the learned evaluator to improve decisions. Search provides the policy-improvement mechanism; simply fitting the same player's outcomes is not itself a guarantee of stronger play.

## Label semantics and pitfalls

- An outcome is one sample of return under the behavior that generated the trajectory, not the optimal value of the state or proof that each action was good.
- A strong decision can lose to an unlucky draw; a weak decision can win with a strong deck. Diverse fights and repeated combat randomness help distinguish quality from luck.
- Many rows from one fight remain correlated observations of a single outcome. Row count is not independent evidence count.
- Exclude off-trajectory child rows: the parent fight's terminal result is not the outcome of those hypothetical continuations. They need separately played continuations to get outcome labels.
- Do not invent terminal labels for capped or otherwise unfinished fights; settle their treatment before implementation.
- Exploratory mistakes affect all earlier states' returns. Those labels are valid for the exploratory behavior, but differ from the desired exploration-free policy. Keep exploration modest initially and make its treatment explicit.
- A terminal-score prediction already includes the expected final HP/potion contribution. Do not add current HP as another reward.
- Keep training/evaluation separation by source run/fight family, including resamples. The repeatedly examined v6 fights are development data for model selection, not a fresh confirmation set.

## Small first experiment

**gen0 → one batch of outcome-labelled fights → one fine-tuned candidate → paired gameplay evaluation.**

No new architecture, policy head, binary objective, or long unattended iteration loop. Dataset size, exploration, replay mix, optimization settings, and promotion criteria remain to be agreed.

A useful control is outcome-only training on existing teacher trajectories. This helps separate changing supervision from collecting gen0's own experience. Compare on the same evaluation cohort; account for differing data quantity and optimization effort before attributing gains.

## Implementation considerations to assess later

- The trainer already has a `terminal` label mode, but `assign_targets` still gives child rows their search estimates. Selecting this mode alone is therefore **not** terminal-only supervision: filter to eligible completed on-trajectory decision rows.
- Review existing fight generation/resampling support for value-net play before adding another app.
- Existing DAgger rows carry learner outcomes as well as teacher labels; assess whether they are reusable as an initial outcome-only dataset, without changing the meaning of the original DAgger runs.
- Decide whether exploration should be off for the first batch or whether exploratory trajectories should be handled specially.

## Related notes

- `../../rundecks/slime-v6/REPORT.md`: DAgger assessment.
- `../../rundecks/slime-v7/PLAN.md`: fair-search assessment before committing to expert iteration.
- `research_axes.md`: broader research questions.

The guiding idea is **less dependence on teacher judgments**, not necessarily less informative rewards.
