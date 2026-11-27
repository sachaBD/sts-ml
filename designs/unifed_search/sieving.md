# Sieving: stop globally, allocate locally

- **Goal:** stop when further search is not expected to improve the **root move** enough to justify its cost. Confidence in the fight value is not the target.
- **Rung** = evals at one action depth. **Sieve** = a stop-or-spend decision using the partial search. Buying compute need not add a rung.

| Decision | Question | Options |
|---|---|---|
| **Global: stop or spend?** | Is the expected root gain from a continuation worth its evals? | Commit the current root move, or continue. |
| **Local: spend where and how?** | Which branch and purchase could affect that choice? | **Depth:** extend a line. **Width:** test another action. **Certainty:** compare current contenders on more shared worlds. |

- **Child `thinking`:** benefit of probing *if acting at that child*. Allocation evidence, not a global stop score or a pruning veto.
- **Root relevance:** consider whether the branch is reached and whether revising its value could change a competitive root move.
- **More worlds:** reduce sampling uncertainty; do not fix a systematically bad continuation.

## Forward-looking target (proposal)

1. **At a probe checkpoint:** save the partial search and the root move `A` we'd play now.
2. **Continue for a specified extra eval budget `b`** under a specified allocation rule. Record root move `B` at that budget, even if intermediate checkpoints still chose `A`.
3. **Label `gain(b)`:** zero if `A == B`; otherwise signed `win(B) - win(A)` from the [fresh-world paired comparison](readme.md). Never choose the best-looking future checkpoint in hindsight.
4. **Predict:** `expected gain(partial search, b)` for a few short and longer budgets. One budget-conditioned scalar output; no separate head per rung required.
5. **Price the options:** compare predicted gain minus compute price times extra evals against **stop = 0**. A longer purchase can pay for setup evals even when the next rung alone would not.

- **Scope:** labels price the continuation actually run, not the best imaginable allocation. A fixed-budget forecast does not yet price the option to stop early.
- **Open:** checkpoint budgets, compute price, and local allocation rule.
