# Overnight card-outcome research runbook

## Goal

Determine whether richer combat data improves outcome prediction—especially the predicted difference between taking a card and skipping it. Ironclad A20, Act 1. Nine-hour total budget, 11 generation workers, GPU available for agents.combat.value.

**Use `generate.sh` and `train.sh` as described in README.md. Each checkpoint is a separate invocation; there is no automatic seven-hour loop. Arrange Opus completion notifications in the supervising harness.**

## Before starting

- User-owned combat-generator preflight is complete.
- Snapshot the executable and record initial generation settings, small network topology and training recipe. Keep topology fixed; Opus may adjust data generation and training settings between stages, documenting changes.
- Establish fixed development/early-stop cohorts, covering easy, hard, elite and boss encounters. Keep all variants and repeated fights from a source group together; keep donor-linked splits consistent with natural data.
- Set an absolute finish time and preserve logs/artifact paths. Schedule an Opus-5-5 wake-up after each train/evaluate checkpoint; do not rely on an agent staying active.

## Stages

| Stage | Work | Checkpoint |
|---|---|---|
| 1 | Generate for approximately **45 minutes**, covering all four encounter categories. Train natural-only and augmented-data models; evaluate. | Wake Opus to review the first results and decide whether to continue. |
| 2 | Generate for approximately **2 more hours**. Retrain on cumulative training data; evaluate on the same held-out cohorts. | Wake Opus to compare progress and decide the final allocation. |
| 3 | Generate for **2–4 more hours**, adjusted to measured costs and time remaining. Retrain on cumulative training data; evaluate. | Wake Opus to write the morning report. |

Generation times exclude agents/combat/value/evaluation. Reserve enough of the nine-hour budget to finish training and reporting; shorten stage 3 if necessary. Persist completed finite batches throughout. Do not spend the entire first stage on easy fights.

## At each wake-up: Opus checklist

1. **Data:** completed groups/fights by category, deaths, exclusions and policy provenance. Check labels and split integrity; retain category-balanced agents.combat.value.
2. **Training:** finite losses, sensible predictions, completed checkpoints, actual runtime.
3. **Evaluation:** compare outcome accuracy by category/encounter and paired card-effect error against predicting zero card effect. Compare the augmented model with the natural-only reference and preceding checkpoints.
4. **Decision:** exercise independent research judgment. Compare easy, hard, elite and boss results and choose the most useful next step—not merely whether to continue the original schedule. Record a short rationale and the next wake-up/artifact location.

## Opus's discretion

Opus is the overnight researcher, not just a monitor. The stages are checkpoints, not a rigid allocation plan. It may:
- Shift generation effort between categories, encounters or underrepresented states.
- Choose more distinct decks versus more repeated seeds, based on coverage and measurement noise.
- Adjust data mixtures, sampling weights, generation settings or training settings when results justify it.
- Run a small diagnostic, investigate an anomaly, fix a straightforward pipeline bug, shorten a stage or stop an unproductive branch.

Use judgment rather than mechanical thresholds. A noisy first result need not mean failure; a clear category-specific problem should influence the next experiment. Prefer changes that answer an identifiable question over broad sweeps. Keep the topology fixed for this run.

Guardrails: stay within the nine-hour/resource budget; preserve raw data, provenance and prior checkpoints; never move held-out groups into training or use reserved confirmation data. Keep the fixed evaluation cohort for comparisons; supplementary diagnostics may be added and labelled. Do not silently change simulator semantics or overwrite completed runs. Pause affected work when data validity or split integrity is uncertain. Document changes and treat repeated development-set inspection as exploratory research, not independent confirmation.

## Morning report

A concise table for each checkpoint:
- Data volume/diversity and compute time by category.
- Outcome prediction accuracy.
- Card-effect accuracy versus zero effect and the natural-only model.
- Whether additional data helped, important uncertainty/failures, and the recommended next experiment.

These are development results, not proof of improved gameplay. Realistic boss-state data does not by itself resolve the earlier low-resource gauntlet mismatch.
