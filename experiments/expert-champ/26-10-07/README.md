# Expert Champ — information-efficient hybrid expert iteration

Status: session scope and proposed research sequence; no gameplay or training launched.

## Objective

Improve the confirmed Demon Form/Pyramid specialist against A20 Champ at a practical single-machine budget. Retain rollout assistance at inference. Long-term objective: a generic combat agent; no card-specific rules or deck-specific architecture.

Agent: **rollout-assisted neural MCTS** (learned priors, PUCT 2k, 50% neural / 50% guided-rollout leaf evaluation). Training: **rollout-assisted expert iteration**.

## Starting evidence

- Frozen Phase5 strong/update3: 485/600 final wins vs MCTS20k 441/600; paired gap +7.33pp, approximate 95% CI +4.62 to +10.04pp. One training RNG and one fixed loadout, 34/52 HP, no potions.
- Model: `runs/schema=combat_v4/date=2026-10-06/id=rollout-expert-iteration-v1/out/strong/update3/model/`.
- Preserve original model, optimizer, frozen worker and evaluation artifacts. Original worker SHA256 prefix: `58c5c4ab`.
- Reference: `experiments/combat-agent-rescue/counterfactual-rescue/FINAL_CONFIRMATION.md`.
- Phase5 already tried fresh full-fight hybrid expert iteration (600 fights, three updates). This session is continuation/improvement, not the first test of that idea.
- Historical teacher correction received 7.5% of policy-loss mass and weakly changed policy. Deliberate fitting worked better, but a fitted prior with fixed original evaluator did not establish stronger play. The 7.5% measurement applies to the old correction run, NOT automatically to the current hybrid trainer.
- Phase3 forced single-action alternatives under the old learner continuation produced weak evidence; no counterfactual-label training followed. Do not repeat this unchanged.
- Existing overnight continuation artifacts exist but no completed REPORT/state was found in the initial lookup. Audit actual coverage and processes before reusing machinery or dispatching jobs; absence of a report is not proof no games ran.

## Proposed two-hour exploration (not a guaranteed runtime)

1. **Audit and freeze:** verify current resource ownership, existing overnight ledger, baseline artifacts and trainer contracts. Separate any Headbutt representation/search fix from this experiment; record worker/encoder hashes and prohibit silent baseline changes.
2. **Cheap absorption audit using training data only:** measure current hybrid-target fit, policy confidence weights, effective source contribution, duplicate/forced-decision exposure and value fit. Hold out whole training fights for diagnostic generalization; do not use final outcomes to train or mine corrections. Diagnose before changing loss weighting.
3. **One bounded learning comparison:** prioritize ordinary hybrid expert iteration as the control. Select ONE data/learning change from the audit, such as explicit exposure to underlearned fresh hybrid targets. Match initialization, data where possible, optimizer steps and fixed endpoint; record when changed weights alter gradient scale. Retain wins, losses and representative replay. Freeze exact intervention/budget before training.
4. **Evaluate full fights with the hybrid retained:** use development starts for cheap screening; promote only against the incumbent on independent paired starts. Improvement in target fit alone is not success. Preserve all checkpoints; no endpoint selection on final results.

If existing hybrid targets are already well absorbed, do NOT intensify fitting by default. Pivot to a bounded teacher-quality check: can stronger hybrid search produce better decisions at learner-visited training states? Stronger compute is not assumed to mean stronger labels. Use public-root sampling before actions and legitimate continuation semantics. Implement targeted reanalysis only if supported by this check.

## Overnight decision

Launch only after explicit configuration/runtime/resource/deadline review. Prefer one reproducible continuation with automatic reports over a broad sweep. Keep a frozen incumbent, monitoring set and untouched final set; predeclare selection and stopping rules. Reserve time for independent confirmation. Report training-seed uncertainty separately from evaluation-seed uncertainty.

## Guardrails

- One compute-heavy stage at a time; at most 10 gameplay workers globally, subject to actual free cores.
- Announce expected runtime (or uncertainty), goal and preserved log path before any >1-minute command; run in background.
- No final/monitor trajectories as training data; audit previously used AND reserved seeds.
- No automatic gameplay retries or replacement seeds. Caps/errors are not losses.
- Results/checkpoints under `runs/`; any new build under `build/<name>/`.
- Do not overwrite unrelated working-tree changes or frozen artifacts.
- Gumbel/sequential halving remains a candidate, not a concurrent implementation.
- Barricade and multi-deck expansion deferred during this session's Demon Form focus.
