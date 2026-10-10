# Targeted teacher correction experiment

User approved investigation/intervention 2026-10-06. No automatic scope expansion.

Hypothesis: learner search reinforces a blind spot (notably Dual Wield favouring
Boomerang over zero-cost attacks); teacher trajectories on failed training starts can
provide corrective examples missing from self-play.

Frozen protocol:
- Start both branches from Demon Form run update15 weights and optimizer.
- Active replay: 1,000 learner fights from rounds6–15; 420 losses.
- Select100 of those losses, uniform without replacement, Random(20261006), sorted IDs.
- MCTS20k replays those exact training starts, all actual outcomes retained.100 games;
  10 workers. Halt if any incomplete; no automatic retries or seed substitutions.
- Control trains existing1000 trajectories; correction substitutes the100 selected
  trajectories with teacher-generated trajectories (retaining900 learner trajectories).
- Three epochs each, lr3e-4,64 states/fight/epoch with replacement, grad clip1, same
  value/policy objectives and AdamW resume. Both use mixed-shard minibatches, a shared
  batching change from the original streaming run. Both start RNG0 and select best
  checkpoint by existing validation-loss rule over three epochs.
- Evaluate each model learner2k on same100 monitoring seeds. Primary comparison:
  correction minus control, paired win-rate gap and uncertainty; also saved update15
  and cached MCTS. Monitoring is consumed/exploratory, not final confirmation.
- No new self-play; no use of monitor/final losses for training; final seeds untouched.
- One checkpoint/recipe/replicate only; no automatic follow-up.

Interpretation limit: complete teacher paths are different state/policy/outcome data;
this does not uniquely identify which supervision component helped. Equal fight and
sample budgets do not ensure identical gradient order (teacher trajectories differ).
No hard-coded card preferences. Outcome shift from replacing losses is part of the
intervention, not proof of a purely policy-label effect.

Preflight: deterministic loss sampling tests + existing runner tests, nine passed.
All active replay seeds verified disjoint from monitor/final.

Run: `runs/schema=combat_v4/date=2026-10-06/id=demon-form-teacher-correction-v1/`.
Live hypothesis/protocol/result runbook: run `out/RUNBOOK.md`.

```sh
watch -n 5 '.venv/bin/python -m apps.run_rl.teacher_correction status --run runs/schema=combat_v4/date=2026-10-06/id=demon-form-teacher-correction-v1'
```

Logs: run `logs/stdout.log`, `logs/stderr.log`; per-stage logs in `out/logs/`,
`out/control/logs/`, `out/correction/logs/`. Results: `out/summary.json`.

## Completed result

Background job3 confirmed exit0 with bg_wait. All100 teacher and200 branch evaluation
fights completed. Teacher recovered44/100 selected training losses (95% Wilson34.7–53.8%).
Control69/100, correction70/100, original update15 66/100, cached MCTS79/100.
Primary correction-minus-control +1pp, paired SE3.89pp; approximate95% normal interval
-6.6 to +8.6pp. One training RNG, consumed100-seed monitor: no convincing evidence of
benefit from this100-trajectory/three-epoch correction dose. Does not rule out stronger
teacher support or teacher labels on learner states. No automatic follow-up launched;
final seeds remain untouched.

## Training-start uptake diagnostic

User approved replay of targeted100 training starts with both branches, greedy2k,
10 workers, no training. Run `demon-form-correction-uptake-v1` completed (bg_wait,
exit0), all200 plays complete. Correction40/100 vs control39/100; paired gap+1pp,
approximate95% interval -7.1 to +9.1pp. On44 teacher-rescued starts correction31/44
vs control29/44 (+4.5pp, approximate95% -11.0 to +20.1pp). On56 teacher-unrescued
starts correction9/56 vs control10/56. No convincing correction-specific gain even
on exposed starts. Original learner losses used exploration, so0/100 is not a
comparable pre-correction baseline. No additional compute launched; final seeds
untouched. Full protocol/result: uptake run `out/RUNBOOK.md` and `summary.json`.
