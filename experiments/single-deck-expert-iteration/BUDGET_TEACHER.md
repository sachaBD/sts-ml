# Matched teacher gap and learned-search budget response (Demon Form, decoupled-dev starts)

Approved by single-fight-astra 2026-10-06. Same 100 development starts (sha `0221e66d…`, byte copy, not
regenerated) as DECOUPLED_POLICY.md. New arms: frozen rollout MCTS teacher 20k (deployed objective,
`pv_worker teacher 20000`) and original update15 at 10k learned simulations (`pv_worker play update15.onnx 10000`;
other settings identical). Cached: update15 2k baseline, decoupled 2k intervention. Same frozen worker
`58c5c4ab…`. Development seeds, not final confirmation; fixed budgets, not equal-wall-time claims.

Run: `runs/schema=combat_v4/date=2026-10-06/id=demon-form-budget-teacher-v1/out`; script `budget_teacher.py`.
Teacher 12:31:04–12:32:25, learned 10k 12:32:25–12:38:55 (+01:00); bg_wait exit 0 (true status propagated);
200/200 completed, no errors/caps/retries; no pv_worker left (timeout process-group kill verified separately).

## Results (intended 100 per arm; 100 completed pairs each)

| Arm | Wins | Wall s/game mean / median |
|---|---:|---:|
| update15 2k (cached) | 62 | 6.58 / 6.46 |
| decoupled policy 2k (cached) | 64 | 8.47 / 8.29 |
| update15 10k | 67 | 34.4 / 34.2 |
| MCTS teacher 20k | 74 | 7.08 / 6.71 |

| Comparison (X − Y) | Gap | approx. 95% | both won | only Y won | only X won | both lost | exact McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10k − 2k | +5 pp | −2.6, +12.6 | 57 | 5 | 10 | 28 | 0.30 |
| teacher − 2k | +12 pp | +2.6, +21.4 | 56 | 6 | 18 | 20 | 0.023 |
| teacher − 10k | +7 pp | −3.1, +17.1 | 57 | 10 | 17 | 16 | 0.25 |
| teacher − decoupled 2k | +10 pp | +1.9, +18.1 | 60 | 4 | 14 | 22 | 0.031 |

(Correction: an earlier version labelled discordance columns "first/second only" ambiguously; report.json keys now
name arms explicitly; old file kept as `report.v1-ambiguous-discordance-keys.json`. Numbers unchanged.)

Intervals conditional on these 100 seeds and the fixed models. The teacher gap on fresh seeds (+12) matches the
monitor estimate (+13). 5× learned budget (5.2× wall time) recovered a non-significant +5.

## Search depth (definition)

Telemetry per searched decision: `mean_turns` = mean over that decision's simulations of turn boundaries crossed
between root and leaf. Per-simulation distribution (all paths, weighted by simulations):
2k: 0 turns 25%, 1 turn 61%, 2 turns 12%, ≥3 2% (5.98 M paths); 10k: 18% / 59% / 18% / 6% (30.3 M).
Per-decision `mean_turns` quantiles 10/25/50/75/90%: 2k 0.08/0.63/0.97/1.16/1.59; 10k 0.18/0.88/1.03/1.42/1.99.
Most simulations cross into the next turn but rarely beyond; leaf values one turn ahead carry most of the
decision. Evaluator dominance remains a hypothesis.

## Calibration (descriptive; first decision is fixed-time, preferred)

Raw update15 value at the first decision: mean 63.0 vs 62 realized 2k wins; first-decision search root value
mean 70.3 (2k) and 75.3 (10k) vs 62/67 realized (optimistic by ~8 pp). Discrimination at the first decision
is weak: AUC raw→2k outcome 0.60, root 2k→2k outcome 0.63, root 10k→10k 0.68 (n=100). By raw-start bucket
(n; wins 2k/10k/teacher): <50 (30; 17/18/21), 50–70 (32; 18/22/20), 70–85 (23; 14/14/20), ≥85 (15; 13/13/13).
Max-over-fight root ≥80 among 2k losses: 33/38 (selection- and length-confounded; descriptive only); 10 of those
33 won at 10k. Teacher won 18 of the 38 2k losses. Aggregate calibration does not establish causal value error.

## Policy-only feasibility (not run)

Frozen worker supports `play MODEL SIMS --policy-only` (argmax raw prior, no search). `apps.human_champ.bench`
does not pass that flag; a tiny Python driver change would be required (no native code). Cost < 1 s/game.
