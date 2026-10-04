# Phase 2 item 1: real-play turn search

`pv_worker play MODEL E --turn-search --particles K` (without `--oracle`), or
`apps/pv/play.py --agent pv --turn-search --particles K --sims E ...`.
Default K=4. This is evaluation-only: no noise, early sampling, rollout mixing,
training targets or policy-only mode. Existing per-action behavior and oracle
turn-search output are unchanged without the new real-turn mode.

At every non-forced real decision, sample K `teacher::public_particles`, and
sequentially search each as a true state with the existing oracle turn search.
Group root children by `pv::action_key` of their first action and take the maximum
Q per group. Average over K particles; a missing action receives that particle's
**network root prediction clamped to [0,100]**, not its max-child value. Execute
the best real legal action. No cross-particle or cross-decision reuse.

A capped particle contributes existing per-action oracle PUCT800 root edge Qs
(unvisited edges use clamped network-root FPU). No partial macro children are
used. Macro caps remain root2048/non-root512, sequences20k, 1s, path512, 256MiB
conservative accounting; V batches ≤32. Rune Dome is still rejected by the public
belief contract, not silently treated as observable.

`particle_stats-*.parquet` records K, elapsed seconds, mean macro leaf depth,
fallback count/reasons, time-cap fallback count, any macro time-overshoot count,
network calls and evaluated states per non-forced decision. Capped particles
have no complete macro tree and contribute zero macro depth; the fallback count
makes this denominator explicit. Forced actions have no search telemetry. No
invented policy/visit rows are emitted by this real-play evaluation mode.

Tests cover max-Q grouping, mean-over-particle aggregation, missing-action root
V, public draw-selection key mapping across hidden order, real legal execution,
per-particle fallback, separate time-cap counting and independence from true
private RNG. Existing macro/native checks, no-flag byte-stability and r10 replay
regression are also run. Builds and smoke use CPU10 only (never CPU11, which is
running turnbench); private build directory `build/pv-phase2` avoids live build
races. Frozen binaries are immutable per launch.

## Item 1 checkpoint

r13, K=4, E=16, first20 nodome starts, CPU10, frozen
`build/frozen/pv_worker.pimc-136e4ebe28f7`:
- **1/20 wins**, 0 capped fights; all complete canonical replays checked.
- 335.90 wall seconds = **16.79 seconds/fight**; decision mean15.67.
  Observed per-fight decision median16.09, p9024.98, range0.19–29.01 seconds
  (n20; this range is workload/run variability, not uncertainty bounds).
- 623 non-forced decisions, 2492 particle searches; 106 fallbacks (4.25%):
  98 child caps,8 sequence caps;0 time fallbacks,0 time overshoots.
- Per-decision macro mean-depth average1.68,p902.60 turns (n623, includes
  capped particles at depth0).

Poor small-slice result; no matched/paired per-action comparison was run and
uncertainty/generalization to the full bench is unknown. No autonomous algorithm
redesign was made. Smoke artifacts:
`runs/schema=combat_v4/date=2026-10-04/id=champ-pimc-r13-k4-e16-smoke/`.
