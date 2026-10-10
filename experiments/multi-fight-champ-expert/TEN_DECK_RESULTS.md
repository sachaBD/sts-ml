# Ten-deck result: update 6

Recorded from the user's supplied results and checked against the saved per-deck report:
`runs/schema=combat_v4/date=2026-10-09/id=champ-ten-test-u6-v1/out/REPORT.md`.
The aggregate below is user-reported; its win totals match the sum of the report rows.
Paired intervals are approximate 95% intervals, in percentage points; not simultaneous intervals.

Checkpoint: `runs/schema=combat_v4/date=2026-10-09/id=champ-ten-rollout-v1/out/update006/model`.
Evaluation: network + 2,000 search simulations, rollout_mix=0.5, versus MCTS20k;
300 fresh paired seeds per deck. This is update **6**, not the originally planned update 10.
The precise reason/timing of that endpoint change has not been audited here.
Training config: 300 teacher fights/deck, then 6 × 200 learner fights/deck;
width64, actual-outcome value targets, root-visit policy targets, 3 epochs/update,
lr 3e-4, exploration and early-turn sampling during collection, 64 states/fight.

## Starting fights and selection screens

| name | deck prefix | type | HP | MCTS20k screen | 95% Wilson | screen source |
|---|---|---|---:|---:|---:|---|
| apparition-block | 51a5aff9 | block | 27/37 | 13/20 (65%) | 43–82% | library |
| perfected-strike | 4f2c64a3 | strength | 56/75 | 6/20 (30%) | 15–52% | library |
| clash | 9308c94d | strength | 59/89 | 5/20 (25%) | 11–47% | library |
| corruption-fnp | 691c2b82 | exhaust | 38/82 | 4/20 (20%) | 8–42% | library |
| fire-breathing | 35a58920 | strength | 48/77 | 4/20 (20%) | 8–42% | library |
| limit-break-rage | 35084002 | strength | 54/75 | 7/20 (35%) | 18–57% | hdc |
| barricade-searing | 879b437e | block | 44/91 | 8/20 (40%) | 22–61% | hdc |
| exhume-immolate | 603cf664 | mixed | 41/109 | 8/20 (40%) | 22–61% | hdc |
| demon-combust | 168ec20d | demon form | 62/80 | 11/20 (55%) | 34–74% | hdc |
| panache | 3a7d5ed7 | exhaust | 43/95 | 12/20 (60%) | 39–78% | hdc |

## Paired test

| deck | model | MCTS20k | model − MCTS, pp (95% CI) |
|---|---:|---:|---:|
| apparition-block | 171/300 (57.0%) | 142/300 (47.3%) | +9.7 [+3.1, +16.3] |
| perfected-strike | 63/300 (21.0%) | 40/300 (13.3%) | +7.7 [+3.1, +12.3] |
| clash | 114/300 (38.0%) | 84/300 (28.0%) | +10.0 [+4.0, +16.0] |
| corruption-fnp | 64/300 (21.3%) | 42/300 (14.0%) | +7.3 [+2.5, +12.2] |
| fire-breathing | 114/300 (38.0%) | 84/300 (28.0%) | +10.0 [+3.9, +16.1] |
| limit-break-rage | 166/300 (55.3%) | 141/300 (47.0%) | +8.3 [+2.4, +14.3] |
| barricade-searing | 292/300 (97.3%) | 152/300 (50.7%) | +46.7 [+40.8, +52.5] |
| exhume-immolate | 137/300 (45.7%) | 91/300 (30.3%) | +15.3 [+8.4, +22.3] |
| demon-combust | 179/300 (59.7%) | 151/300 (50.3%) | +9.3 [+3.3, +15.4] |
| panache | 174/300 (58.0%) | 124/300 (41.3%) | +16.7 [+10.3, +23.1] |
| all | 1474/3000 (49.1%) | 1051/3000 (35.0%) | +14.1 [+12.2, +16.0] |

All ten point estimates favor the model. Excluding barricade-searing, the pooled gain is
283/2700 = +10.5 pp (interval not computed here), so the aggregate is not solely that deck.

Scope: one shared model, ten selected **training decks**, fresh fight seeds, one training run.
This supports seed generalization on those starts, not unseen-deck superiority. The aggregate
interval is not an uncertainty interval over the population of human decks or training RNGs.
Twenty-seed selection screens are noisy and selected; do not interpret their difference from
the 300-seed references as a temporal regression. Raw outcomes and seed independence were
not independently re-audited in this logging pass.
