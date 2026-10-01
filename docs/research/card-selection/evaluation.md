# Evaluation: iteration first

Adapted from `../evaluation_methodology.md`. That document concerns combat-agent comparisons; its sample-size estimates and HP-equivalent score are NOT automatically appropriate for surrogate evaluation.

## 1. Existing-data validation: required first report

Freeze a source-run manifest and deterministic split by original `run_seed`. All fights, replays and perturbations from a source seed stay together. Existing buckets 0–1 are development combat benchmarks; 2–3 are described as confirmation in the combat notes. Coordinate before consuming those sets; do not assume previously used benchmarks are fresh. Fit preprocessing on training only.

Compare one small model against an encounter/start-HP-aware simple baseline on exactly the same held-out fights. Report:

- Distinct source runs, fights, wins/deaths, encounter counts, policy mix and exclusions.
- Survival: Brier score, log loss, and a small reliability table (predicted versus observed win rate with counts).
- On wins: ending-HP MAE for the predicted mean; histogram NLL and a few threshold probabilities or interval-coverage checks for the distribution. Fix binning across compared models.
- By encounter, with elites and each boss explicit; overall aggregates can hide the failure cases. Include starting-HP bands if sample sizes support them.
- Batch inference throughput on stated hardware/batch size. This is the intended computational advantage.

One observed result per state is valid probabilistic supervision; it is not a ground-truth per-state win probability. Do not demand exact prediction of stochastic outcomes. A repeated-fight subset, if already available, can additionally check calibration without generating new data.

For ordinary iteration show counts and descriptive paired deltas. For material comparative claims use paired bootstrap intervals clustered by original run seed. No mandatory hypothesis-test gate or huge confirmation run for each model change.

## 2. Decision-facing check: small and targeted

Once the first model works, reuse the gauntlet's paired option evaluations where possible. First check whether existing artifacts contain reconstructable candidate inputs; current result rows alone are not a complete input encoding.

Hold reward state, encounter, augmentation procedure and future-offer/RNG sample fixed across offered cards and skip. Compare model and real agent on:

- Pairwise differences in win probability and expected surviving HP (report separately).
- Error in the gauntlet score difference, if using that score as the application objective.
- Real-agent score regret of the model-selected option versus the best estimated option; label finite-sample winner bias and uncertainty.

Do not report noisy best-card labels as unquestionable truth. Start with available matched repetitions; if new runs are needed, use a small pilot and more repetitions only where necessary. Different search salts measure search variability; distinct fight seeds are also needed for the pre-combat RNG-marginal target.

Card differences, not just global outcome accuracy, are the key decision-facing evidence. Keep future augmentation identical in the surrogate and real-agent comparison so this evaluates the surrogate rather than two different gauntlets.

## 3. Later, not an initial blocker

A better predictor/gauntlet ranking does not establish better Act 1 clear rate. After integration, compare actual card policies on paired fresh run seeds with the real combat agent held fixed. Full-run evaluation is an occasional policy check, not the training teacher for every reward.

## Minimum saved report

Input lineage + split; exact policy and target boundary; architecture/output definition; counts/exclusions; baseline and model metrics by encounter; runtime; known missing state. Add paired-card results only when performed. No invented sample sizes or universal acceptance threshold.
