# Slime Boss value-search: research map

## Goal

Long-term: strong A20 Ironclad play. Current instrumental goal: improve on rollout MCTS in Slime Boss combat, where split timing tests nonlinear, delayed consequences.

Research question: **Can a compact learned value function provide evaluations that make search play better than its rollout teacher?**

We are early in exploration. Prefer a few informative experiments over broad sweeps, premature significance claims, or numerical promotion thresholds.

## Five research axes

| Area | Main choices | Central question |
|---|---|---|
| Training data | Amount, deck/state diversity, teacher strength, trajectories versus alternative/search-leaf states | Are we showing it the situations it must judge? |
| Learning target / objective | Terminal outcomes versus search estimates and their blend; value regression versus action ranking; win/HP/potion trade-offs | Are we teaching the right distinctions? |
| State representation and network | Information encoded, card/monster interactions, pooling versus attention, width/depth | Can it distinguish strategically different positions? |
| Optimisation | Learning rate, epochs, batch size, regularisation, sampling weights | Can we reliably fit the examples? |
| Search strategy | Immediate NN versus hybrid rollout, simulation/time budget, rollout horizon, exploration, belief particles, inference batching | Can search turn predictions into better decisions? |

These are research axes, not one giant hyperparameter sweep. Choose an axis based on a plausible bottleneck and change one meaningful factor.

## Current reference setup

The slime-value-4 baseline uses:
- Width-64 DeepSets network: card/monster embeddings and MLPs, summed card-zone representations, explicit card–monster interaction features, value head.
- Blended supervision: equal search-estimate and terminal-outcome weights where applicable. Off-trajectory child rows use search estimates; decisions at/before the exploratory move use search estimates in blend mode.
- 20 epochs; batch 128; learning rate 0.001; weight decay 0.0001; training seed 0.
- MCTS: up to 15,000 simulations, eight belief particles, early stopping; neural batch 64.
- Immediate NN leaves. The initial hybrid ablation uses at most one turn increment / 16 rollout actions before NN evaluation—not one complete future player turn.

Recorded settings/checkpoint metadata remain authoritative for each actual run.

## What we have learned so far

A 14-fight development pilot compared rollout MCTS (R), immediate NN (N), and hybrid (H), all without forced exploratory actions:

| Player | Wins | Mean terminal score | Mean worker time |
|---|---:|---:|---:|
| R | 14/14 | 0.481 | 7.9 s |
| N | 11/14 | 0.352 | 10.0 s |
| H | 12/14 | 0.382 | 11.9 s |

Times include replay of preceding fights. Equal simulations are not equal compute. This small development sample identifies weaknesses; it does not establish general superiority or a reliable hybrid advantage.

Trajectory inspection found poor attack-versus-wait decisions and costly split sizes, but also a neural success using an early potion and a strong split. Thus “always split earlier” or “always delay” is not a sufficient diagnosis. Limited training coverage is a leading hypothesis, not a demonstrated exclusive cause.

Sources: `scratch/slime_pilot14_diag/report.md`, registered `slime-value-4-pilot14-{r,n,h}` runs, and `slop_docs/slime/value_search_experiment_plan.md` for historical experiment details.

## Current experiment: more teacher data

The original model fit only 55 fights. The next experiment increases training data while retaining the basic model, target, and search approach.

Snapshot planning found 450 complete Slime fights: 359 eligible for training/internal validation, 14 original gameplay-development fights excluded, and 77 fresh fights reserved untouched. The existing trainer internally splits the eligible pool, giving approximately 287 fitting fights and 72 validation fights. Check actual checkpoint metadata rather than treating these planning counts as verified run results.

**Status:** user reports `slime-value-5` is trained and will run value play shortly. Its exact input/split/config metadata and gameplay results have not yet been audited here. Intercom agents are offline; no further delegated jobs are pending from this document.

Important interpretation controls:
- Verify which fights value_play selects. By default it uses the new checkpoint's internal validation IDs; those may differ from the old 14 fights.
- Do not compare raw win rates across different fight cohorts as evidence that one checkpoint improved.
- A same-fight teacher comparison is useful on a new cohort; direct old/new improvement requires both checkpoints on matched fights.
- Keep forced exploration off for clean policy comparisons, and use a fresh exploration-off teacher baseline rather than stored exploratory generation outcomes.
- Do not mix gameplay-development or untouched test seeds into training.
- Same epochs with more data also means more optimisation updates: this is a practical data-scaling experiment, not a perfect isolation of data quantity.

## How results guide the next experiment

### Clear improvement

Keep the basic approach. Build a learning curve as data grows before redesigning the network. Check that gains reflect better behaviour, not merely lower regression loss.

### Little improvement

Do not assume more ordinary trajectories will solve it indefinitely. Check whether the network can fit the task and whether the encoding contains necessary information. Then test one change to supervision, training-state coverage, or representation.

### Worse or erratic

First check data consistency, training stability, and evaluation comparability. One disappointing run does not invalidate neural value functions.

### Architecture specifically

The useful question is not “would a transformer be better?” It is: **can this model distinguish states requiring different decisions, and can our labels teach those distinctions?**

A compact pooled network can represent nonlinear effects; missing attention is not proof that it cannot learn split timing. Conversely, a larger model cannot recover omitted information. Capacity changes should answer a demonstrated learning bottleneck, not substitute for checking targets and coverage.

## Priority order

1. Data scale: current experiment.
2. Targets and training-state coverage if scaling disappoints.
3. Representation and capacity when evidence points to a model bottleneck.
4. Search integration revisited with a more reliable evaluator.

Optimisation receives basic sanity checks throughout. Preserve evaluation comparability and report runtime, but avoid turning every exploratory run into a large statistical campaign.

## Working agreement

Coordinator focuses on hypotheses, experimental choices, and concise interpretation. Discuss concrete code changes with the owner before delegation. Routine execution details should be batched, not narrated continuously. No automatic large confirmation ladder or arbitrary numeric promotion gates.
