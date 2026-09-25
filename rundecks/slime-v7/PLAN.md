# slime-v7: fair-search limits before expert iteration

Status: **direction agreed; implementation and exact run settings pending.** No experiments launched.

## Goal

Long-term: expert iteration (search → improved training data → retrain → stronger search).

Next milestone: establish whether a stronger **public-information search expert** can outperform the v6 guided-rollout teacher. Do not start another training generation until we understand whether search supplies an improvement signal.

This rundeck covers search assessment first. Training is a conditional follow-up, not an approved execution step.

## Evidence and interpretation

See `../slime-v6/REPORT.md`, `PLAN.md`, and `LOG.md`.

On the same 156 held-out Slime Boss fights:

| Agent | Wins | Mean terminal value |
|---|---:|---:|
| Guided-rollout teacher | 121 | 0.427 |
| gen0 | 115 | 0.401 |
| DAgger gen3 | 106 | 0.382 |
| Perfect-information oracle, max backup | 152 | 0.571 |

- DAgger's scalar teacher-value corrections on learner trajectories did not improve play. Retain gen0 as the neural reference.
- This does not rule out a coverage problem at speculative search leaves: learner decision states and queried leaf states are different distributions.
- `root_value` averages explored root edges by visits, including exploratory actions. It is not an independent estimate of the selected policy's return. DAgger uses root-only labels; bootstrap partly blends terminal outcomes. Coverage and supervision mix therefore both changed.
- The gen0–teacher difference is not statistically significant, but equivalence is not established.
- The oracle changes information access, particle count, backup rule, and early stopping. Its achieved score shows perfect-information headroom, not recoverable fair-play improvement or a certified optimal bound. Do not attribute the entire gap to hidden information.

## Questions

1. Is fair teacher strength limited by tree-search effort, uncertainty sampling, or both?
2. Can the frozen gen0 evaluator improve the expert, either directly or after a short guided rollout?
3. Does any gain survive a practical wall-clock comparison?

More simulations over the same eight sampled hidden worlds might optimize a poor uncertainty approximation more thoroughly. More particles without more simulations might instead spread search too thin. Test these separately and together.

## Stage A: search/particle limits, no learning

Proposed starting grid; finalize budgets after implementation review and a runtime smoke test:

| Arm | Leaf evaluator | Simulation cap | Belief particles |
|---|---|---:|---:|
| R00 | Guided rollout | 15,000 | 8 |
| R10 | Guided rollout | 60,000 | 8 |
| R01 | Guided rollout | 15,000 | 32 |
| R11 | Guided rollout | 60,000 | 32 |

Controls:
- Public information only; oracle off; mean backup and normal most-visited action selection.
- Same fights, starting states, reward, action limits, and particle-sampling scheme.
- Random exploratory move off.
- Preserve baseline early stopping initially; report actual simulations as well as the cap. Explicitly settle forced-action budgets and early-stop semantics before execution.
- Reproduce R00 against the recorded v6 teacher before interpreting new arms.
- Do not change rollout heuristics, train a model, or tune architecture in this stage.

Readout: paired terminal-value differences versus R00, paired win/loss swaps, runtime, and actual search work. Inspect whether the joint arm behaves differently from either change alone. A flat result at one larger budget does not establish a hard ceiling; decide whether a further limit probe is informative after seeing it, rather than precommitting to a broad sweep.

## Stage B: can gen0 strengthen search?

At an informative setting from Stage A, compare:

1. Full guided rollout.
2. Immediate frozen gen0 value evaluation.
3. Bounded guided rollout followed by frozen gen0 evaluation.

Hybrid support already exists in `agents/teacher_leaves.cpp`. The previous 14-fight hybrid pilot used an older model and is not decisive for gen0. Select one explicit horizon before running; verify what a turn increment means rather than calling it a complete future player turn.

First compare equal simulation caps for diagnosis. Then compare promising candidates at comparable wall-clock cost. Include the original 15k/8 teacher and gen0 references in reporting. Equal simulations are not equal compute.

## Evaluation and safeguards

- Use v6's fixed 156 fights as a **development benchmark**, not a new untouched test set. They have already influenced model selection.
- Freeze gen0 and fight selection across arms; record exact model/run IDs in configs once resolved.
- Primary outcome: mean terminal value, with paired differences and confidence intervals. Secondary: wins, win/loss swaps, HP and potion outcomes, runtime, actual simulations, and incomplete fights.
- Terminal value rewards survival, HP, and potions; do not silently substitute win rate as the optimization objective.
- Treat exploratory comparisons as exploratory. No significance fishing or automatic promotion ladder.
- Reserve fresh fights for confirmation after choosing a candidate. Split by source run/fight family so resamples cannot leak across training and evaluation. Confirm the selection rules before acquiring the final cohort.
- If seed sensitivity matters, distinguish search-sampling randomness from fight randomness; repeated identical deterministic searches are not independent evidence.
- Keep search-setting metadata sufficient to reproduce every arm. Report any deviations from this plan.
- One substantial job with 10 workers at a time, following v6's resource guardrail unless explicitly revised.

## Decision after search assessment

**If a stronger fair expert emerges:** confirm it, then design one refinement generation on training fights only. Compare equal quantities of additional original-teacher data and improved-expert data, holding representation and training recipe fixed initially. Evaluate the apprentice at its intended deployment budget. This separates expert quality from data quantity. Avoid simultaneously adding a policy head, new targets, and a new architecture.

**If only extra compute improves play:** decide whether an expensive offline expert is acceptable and whether its gains can be distilled into the desired deployment budget.

**If no tested search variant improves:** investigate decision-target quality and speculative-leaf coverage before another self-training loop. Do not interpret a null result as proof that expert iteration cannot work.

## Next session step: implementation planning

Before delegating code changes:

1. Audit where simulations, particles, early stopping, and forced-action budgets are fixed; identify the smallest explicit configuration surface needed by `value_play` and the shared search helpers.
2. Check fair sampling, reproducibility, and baseline compatibility when particle count changes.
3. Decide which runtime/search-work diagnostics already exist and which are required.
4. Define tests: unchanged defaults reproduce baseline behavior; custom settings reach search and metadata; invalid settings fail; fair search never enables oracle/max backup accidentally.
5. Agree exact Stage A settings and a small smoke-test procedure, then hand a bounded implementation task to an agent.
6. Review the implementation and smoke-test outputs before launching limit testing.

No implementation or agent delegation has been authorized by this document alone.

## Open decisions

- Acceptable offline-expert compute and desired deployment latency.
- Exact simulation/particle limits after runtime inspection.
- Early-stop and forced-action-budget treatment across arms.
- Hybrid horizon for Stage B.
- Fresh confirmation cohort size and selection.

## Files

Configs will go in `configs/` after implementation and input IDs are settled. Execution history goes in `LOG.md`; results and conclusions in `REPORT.md`. Do not overwrite the v6 assessment.
