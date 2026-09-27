# Act 1 card selection

Planning notes, not an implemented system or committed experiment protocol.

## Goal

Ultimately: maximize Ironclad A20 Heart win probability.

First bounded problem: improve ordinary card-reward choices during Act 1, with a **frozen combat agent** and **fixed controllers for every other decision**.

Act 1 boss survival is a provisional development objective, not proof of better full-game play. Optimizing it can favor decks that fail later.

## Core idea

At a card reward, compare taking each offered card with skipping:

1. Copy the same pre-choice run state.
2. Apply one candidate choice.
3. Continue with fixed downstream controllers.
4. Compare success across sampled futures.
5. Learn a selector from useful comparisons.

Actual combat is the reference evaluator, but expensive. A learned combat-outcome surrogate might make these continuations cheap enough to scale. It must earn that role by preserving **card-choice quality**, not merely fitting historical HP loss.

## Current direction vs open questions

**Direction:** learn card rewards only; keep combat fixed during a selection-learning cycle; retain the effects of other game systems under fixed controllers.

**Proposals, not settled requirements:** start with real Act 1; use boss survival as the initial endpoint; accelerate rollouts with a combat surrogate; compare root choices before implementing full strategic MCTS.

**Not prerequisites:** deeper combat networks, learned routing/events/shops, joint combat-and-drafting updates, or full-game neural simulation.

## Visual learning lab

Open [explainer.html](explainer.html) directly in a browser. This implementation workshop covers a concrete outcome-model architecture, completed-fight labels, probability losses, reliability tests, and rollout integration. Its interactive lab actually trains a small categorical model; change data coverage, sample count, and representation, then inspect predicted distributions and sampled continuations. All lab data are synthetic, not agent results. The architecture is a proposed baseline, not implemented repository code.

## Read next

- [COMPONENTS.md](COMPONENTS.md): the small problems and how they connect.
- [PLAN.md](PLAN.md): bounded experiments, artifacts, and success gates.
- [Combat results](../act-1-combat/2026-09-26-mirror-mcts/REPORT.md): the existing shared combat foundation.

## What would count as success?

A new selector improves held-out **actual played-through Act 1 outcomes**, under identical downstream controllers and an explicit compute budget. Surrogate-predicted improvement alone does not count.

Before substantial optimization, check longer-horizon outcomes for evidence that Act 1 gains harm later play. Full-run improvement remains a separate claim.
