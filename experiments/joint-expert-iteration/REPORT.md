# Joint expert iteration — wrap-up

**Stopped at the user's request on 2026-10-02, 07:22 UTC. No experiment processes remain.** Final baseline comparison intentionally cancelled; partial outputs preserved, not analyzed as a complete comparison. Launcher exit status records the interruption rather than successful completion.

## Results

- Three staggered combat/overworld training rounds completed.
- First overworld update promoted on 400 paired development seeds: **92.25% vs 86.75%, +5.50 ±1.89 percentage points** (one paired standard error). Later overworld updates did not pass the gate.
- None of three combat updates passed the gate; their development gains (+0.5 to +1.0 pp, n=400 each) were within noise. Combat remains `ab-gen1`.
- Selected pair completed 800 fresh runs: **709 clears, 88.625%**. No completed fresh baseline comparison, so improvement over the starting pair remains **unconfirmed**.
- Initial neural combat trailed rollout combat on 400 paired development seeds: **88.25% vs 92.25%, −4.00 ±1.53 pp SE**. This is not a comparison of the final pair against rollout combat.

## Retained checkpoints

- Combat: `value_net_v1/2026-09-27/ab-gen1`.
- Overworld candidate: `value_net_v1/2026-10-02/joint-1001-overworld-r00/out/model.pt`.
- All other candidates, logs, manifests and partial evaluations retained. Detailed gates are in controller `out/state.json` and [RUNBOOK.md](RUNBOOK.md).

## Data and engineering

**10,500 exploratory runs; 87,533 fights; 87,025 replay-verified (99.42%); 1,354,296 recorded combat decisions.** 508 unsupported initial callback boundaries retain results only, excluded from replay facts and combat training. 77 collection fights used the turn-30 rollout safeguard. Last 1,500-run batch remains available for future training.

Added replay-first `combat_v4` recording and public macro-trace `overworld_v1`, with separate agent annotations and disposable NN encoding caches. Native replay test passed 21 fights; both training/export smoke paths passed; 29 scoped Python tests and whitespace checks passed at wrap-up.

## Scope / handoff

Ironclad A20 **Act 1 only**; no full-game improvement claim. Easy combat: 500 rollout sims; hard/elite/event/boss: neural leaves 5k/10k/10k/20k, eight particles. Neural defensive stalls switch to 5k rollout at turn 30. Training: one seeded random combat move within first 24 decisions; overworld epsilon 0.1. Evaluation: neither.

Tonight used the older all-encounter **v2 topology**, not existing **v3 policy/value**. Next phase should reuse/test `elite-v3-t2` (`value_net_v1/2026-09-27/elite-v3-t2`) rather than rebuild it. Exploratory terminal labels need a policy-consistency ablation. Multi-table compaction remains deferred; outputs preserved. Source changes remain uncommitted for review. No further work scheduled.
