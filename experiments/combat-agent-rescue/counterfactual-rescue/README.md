# Combat rescue and rollout-assisted expert iteration

## Current status — 2026-10-06, 23:40 BST

**Candidate: 83/100 development wins versus MCTS 74/100. Fresh confirmation pending.**
The candidate combines a learned policy/value network with 2k PUCT search and 50%
completed-rollout leaf evaluation. No card-specific bonuses or bans were introduced.

| Phase | Result | Status |
|---|---|---|
| Restart boundary screen | Learner late positions easier than early | Complete |
| Controlled backward-restart training | Rescue 66/100 vs control 61/100 | Inconclusive gain |
| Forced single-action exploration | Four promising pairs; weak fresh action evidence | Complete; no training |
| Rollout-assisted leaf evaluation | 76/100 vs network-only 66/100 | Clear development gain |
| Matched fresh-fight expert iteration | Strong endpoint 83/100; training control 80/100 | Promising candidate |
| Frozen candidate vs MCTS on 600 reserved final starts | 485/600 vs 441/600, +7.3pp (95% CI +4.6 to +10.0) | **Confirmed** ([FINAL_CONFIRMATION.md](FINAL_CONFIRMATION.md)) |
| Bounded overnight continuation, two replicas | Pending | Script/SRE preparation |

All earlier full-fight comparisons used the same 100 development starts. Do not call
83 vs 74 an untouched-test result. The original reserved final 600 seeds were audited unused and are now CONSUMED by the
final confirmation (2026-10-06 22:42–22:57 UTC); see FINAL_CONFIRMATION.md.

## Read first

- **[RESULTS.md](RESULTS.md)** — consolidated evidence, recipe, costs, limitations and model paths.
- **[OVERNIGHT.md](OVERNIGHT.md)** — frozen unattended training/selection/confirmation protocol;
  absolute deadline 2026-10-07 07:15 BST.
- [PLAN.md](PLAN.md) — original rescue plan and constraints.
- [LITERATURE.md](LITERATURE.md) — conceptual literature table and references.

Detailed phase records:
[Stage 1](STAGE1.md), [Stage 2](stage2/PROTOCOL.md),
[Phase 3](phase3/PROTOCOL.md), [Phase 4](phase4/README.md),
[Phase 5](phase5/README.md).

## Ownership and execution

Astra owns research design and interpretation. `single-fight-opus` completes the frozen
candidate confirmation; `orch-26-10-6` takes over bounded overnight execution and SRE.
Routine operations are script-driven. SRE escalates research changes and unfixable faults.

Max 10 gameplay workers globally; no simultaneous confirmation and overnight gameplay.
No silent game retries, seed substitutions, checkpoint selection on final results or
changes to frozen original artifacts. Resource/deadline guards and automatic reports.

## Main artifacts

Under `runs/schema=combat_v4/date=2026-10-06/`:

- Candidate: `id=rollout-expert-iteration-v1/out/strong/update3/model/`
- Candidate report: `id=rollout-expert-iteration-v1/out/REPORT.md`
- Confirmation: `id=demon-form-rollout-confirmation-v1/`
- Overnight: `id=overnight-rollout-v1/`

Inference command (existing JSONL combat-start interface):
`pv_worker play MODEL.onnx 2000 --rollout-mix 0.5`.
Use the recorded frozen worker (SHA256 prefix `58c5c4ab`), not an arbitrary current build.

This document is a timestamped status snapshot. Run logs and atomic state files are the
live source of operational status. Results must distinguish development from confirmation.
