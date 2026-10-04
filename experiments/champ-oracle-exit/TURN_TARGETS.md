# Phase 2 item 2: turn-search training targets

Opt in with `--oracle --turn-search --turn-targets`, E≥2. `--explore` (root
Dirichlet α0.3, fraction0.25) and `--sample-turns` (first2 turns, chosen root
child ∝ visits) are allowed only with this gate. Eval-only turn search is
unchanged without `--turn-targets`. Real PIMC is still evaluation-only.

Before every action of the chosen macro sequence, collect visited root children
whose raw sequence starts with the exact played prefix. Group their next action
by the PV public action key, sum visits, and map back to the canonical legal
menu. The row root_value is consistent-child visit-weighted Q, clamped[0,100];
child values are group-weighted Q. Zero mass is an error, never a fabricated
uniform target. Every macro decision, including forced actions and mandatory
END_TURN card choices, gets a normal search row. Root fallback retains normal
per-action PUCT800 rows/noise/early sampling. No per-action macro wall times are
invented: macro telemetry stays in `turn_stats`.

The chosen root child is sampled before the plan is executed; the entire sampled
plan is played and exact-key subtree reuse continues at the next turn. Noise is
applied once when a node becomes the live root, not repeatedly during traversal.
E1 remains valid for evaluation but is rejected for targets because a fresh E1
root has no child visits.

Native tests verify prefix consistency, visit sums, group/row weighted Q,
forced-action rows, value clamping, zero-mass rejection, sampled-child positive
visits and normalized/nonuniform noisy priors. Integration tests verify E1
rejection, unchanged eval-turn output versus immutable pre-target worker,
complete replay and policy-bearing rows accepted by the unchanged native encoder.

## Item 2 checkpoint

CPU10, private build; four native tests and seven worker regression/integration
tests pass. Immutable worker `build/frozen/pv_worker.turn-targets-dc8f9579d2b6`.
20 first-nodome fights, r13, E64, targets+noise+early sampling:
- 5/20 wins,0 capped;76.23 wall seconds (3.81/fight),3.58 decision seconds/fight.
- 165 macro turns,23 fallback (13.94%),all child caps;0 time fallbacks/overshoots.
- Unchanged `agents.combat.pv.data.collect` encodes all20 complete replays;
  814 rows,806 with policy targets. The8 other rows are normal forced decisions
  in legacy fallback turns, not fabricated macro targets.

Smoke is an acceptance/workload check, not a paired/full-bench outcome claim;
uncertainty/generalization unknown. Artifacts:
`runs/schema=combat_v4/date=2026-10-04/id=champ-turn-targets-r13-smoke/`.

## Tag-d driver flags

`exit.py --selfplay-flags "--turn-search --turn-targets"
--oracle-bench-flags "--turn-search" --sims 64
--worker build/frozen/pv_worker.turn-targets-dc8f9579d2b6`.
The three options selfplay-flags/oracle-bench-flags/real-flags are shlex-split and
appended only to their respective play.py calls, defaultempty. Real bench keeps
base2000 sims; policy bench unchanged. Every stage uses the same immutable driver
snapshot of the selected worker, including encode. Quoted single-option strings
are accepted as well as multi-option strings. Malformed quoting fails before
snapshotting or stage side effects. Three mocked command-construction tests pass;
no tag-d stages were launched by this change.

Frozen worker SHA256
`dc8f9579d2b651aa3dc7174bdb33ce0aca498f28b8913c730fc576f2ad38b2b6`
was built from clean simulator648229bf49ec5898ceff3e6dca60a07abee86f28;
private CMake cache resolves `/home/sborowsk/project/sts_lightspeed`.
It includes the Liquid Memories forZeroCost fix; its two-start r10 history
regression passed in `turn-targets-phase2-worker-tests.log`.
