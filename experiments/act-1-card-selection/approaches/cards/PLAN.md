# Bounded experiments and decision gates

Proposed sequence. Each stage should leave a useful artifact even if the next stage is abandoned. No implementation is implied by these notes.

## 0. Fix the experiment contract

**Small question:** exactly what are we trying to improve, against what baseline?

Record the combat checkpoint/search budget, inherited selector, other controllers, endpoint, seed distribution, available observations, and compute allowance. Prefer real Act 1 with fixed non-card controllers if the harness supports it; a combat chain is a testbed, not an equivalent task.

Split by source run/seed before generating branches. Keep all branches, resamples, and related snapshots together. Separate development from untouched confirmation data.

**Artifact:** a short frozen protocol.

**Done when:** two implementations would agree on what information is available and what counts as success. Numerical promotion thresholds are chosen before evaluating candidates, not invented after results.

## 1. Establish the real branching reference

**Small question:** can we correctly compare alternatives at one reward?

Build on simulator snapshots and the existing controllers. Begin with a small selection of late-Act-1 rewards: short continuations reduce cost. Test take/skip, reproducibility, branch isolation, and continuation to victory/death.

**Artifact:** reward snapshots, per-choice real outcomes, and measured seconds per continuation.

**Done when:** mechanics are reliable and the cost of obtaining useful comparisons is understood. Late-act results establish the pipeline, not general drafting quality.

## 2. Test whether card choices have a learnable signal

**Small question:** at an affordable sampling budget, can we distinguish any useful alternatives?

Repeat real continuations from matched reward states. Examine survival differences and uncertainty, including disagreements with the inherited selector. Include representative states as well as deliberately informative ones; label those groups separately.

**Artifact:** a small gold-standard comparison set, including unresolved comparisons.

**Success:** some reproducible, meaningful choice differences can be detected at acceptable cost. If not, first distinguish low decision impact from high variance; do not assume a bigger selector fixes either.

## 3. Test the combat shortcut separately

**Small question:** can a surrogate replace real combat without misleading card comparisons?

Train an initial combat-outcome model from existing outcomes plus targeted branches. Evaluate on held-out source runs and changed decks. Check death calibration, survivor outcomes, runtime, and ranking agreement against stage 2.

**Artifact:** a surrogate checkpoint and a compact error/ranking/cost report.

**Success:** the shortcut buys enough throughput while keeping real-outcome decision regret acceptably small. Set the acceptable trade-off in the protocol after the cost pilot, before the held-out comparison.

**Failure path:** keep real continuations and generate fewer, better-targeted labels. The surrogate is optional infrastructure, not the objective.

## 4. Demonstrate one-step card-choice improvement

**Small question:** does the rollout evaluator recommend better current choices than the inherited selector?

On held-out reward states, let each choose the current card, then use the *same fixed inherited policy* for all subsequent rewards. Validate recommendations with real continuations, even if a surrogate produced them.

**Artifact:** paired real-outcome comparison, with uncertainty and decision cost.

**Success:** evidence of improved endpoint success under this one-decision intervention. Predicted gains that disappear in real continuations are a model/planning failure, not a selector-training problem.

## 5. Distill, then test repeated use

**Small question:** can a fast learned selector retain the improvement, and does it survive making every reward choice?

Train from rollout comparisons. First check real one-decision performance; then play complete Act 1 runs with the new selector at every ordinary card reward. Keep combat and all other controllers unchanged.

**Artifact:** selector checkpoint and paired seed-level Act 1 evaluation.

**Success:** real Act 1 improvement at the chosen compute budget on held-out seeds, with confirmation after candidate selection. Track post-boss states and later-act outcomes where feasible to expose short-horizon harm.

One-step improvement does not guarantee repeated-use improvement: the selector changes its own future state distribution.

## 6. Iterate only after the loop works

Promote a successful selector as the new continuation policy. Collect and audit its new deck distribution. Keep a stable reference set alongside fresh on-policy evaluation.

Refresh combat separately when warranted; then revalidate the surrogate. Consider deeper strategic search only if root rollouts show a concrete limitation. Extend the evaluation horizon before heavily optimizing the Act-1-only objective.

## Common evaluation rules

- Primary evidence: actual endpoint success, not surrogate scores or imitation accuracy.
- Diagnostics: deaths by encounter, HP/potions, choice disagreements, uncertainty, and runtime. These explain results; they do not replace the objective.
- Pair alternatives by reward state and full-run candidates by seed. Account for shared source runs when estimating uncertainty.
- Fix per-method compute budgets explicitly. Adaptive sampling can focus on close choices, but keep independent evaluation to avoid selecting noise.
- Use only public information. Simulator cloning is an experimental tool, not permission for the agent to inspect hidden futures.

## Immediate smallest useful deliverable

**A trustworthy real-continuation comparison for a few late-Act-1 card rewards, with runtime and uncertainty recorded.**

This measures the problem before committing to either new model. It becomes the test fixture for the surrogate, rollout evaluator, and learned selector.
