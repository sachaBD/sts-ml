> **Superseded (2026-10-01):** the active approach is `slop_docs/run-rl/README.md` (run-level RL on real games). Kept for the record.

# Card selection research

## Agreed scope

- Ironclad A20, Act 1. Initial objective: Act 1 clear probability, not eventual run victory.
- Card value is the marginal effect of taking it versus the other offers and skip, conditional on the whole run state and future adaptive play.
- Combat outcomes are relative to a specified combat policy and compute budget.
- Build a learned combat outcome/transition model. Full continuations using the real combat agent are not computationally practical for card evaluation.
- Reuse existing fight data. Keep iteration small; do not require a full-run study before training a first useful model.

## Working direction (not established results)

Use explicit macro rules and approximate the expensive combat transition. First application: the existing gauntlet. Later possibility: cheap macro continuations. Neither the gauntlet's weighted score nor surrogate planning is established as the best card selector.

The central unresolved question: can a model predict the **change caused by a card choice**, rather than merely distinguish generally strong and weak decks?

## Documents

- [Combat model brief](combat-model.md): implementation scope, representation, evidence and limitations.
- [Evaluation](evaluation.md): lightweight iteration metrics and decision-facing checks.
- [Open questions](questions.md): decisions needed from the user or implementation work.

Keep experimental measurements with their run artifacts; link summaries here. Separate observations, hypotheses and decisions. Do not turn tentative ideas into requirements without evidence.
