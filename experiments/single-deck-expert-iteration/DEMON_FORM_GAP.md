# Demon Form/Pyramid gap investigation

Read-only investigation of completed update-15 monitor trajectories. No new search,
gameplay or training, no final seeds consumed. Both sets of 100 recorded fights were
replayed/encoded with the run's frozen worker (200/200 successful). This establishes
replay consistency, not full simulator correctness.

## Paired outcomes (100 fixed monitor seeds)

65 both won; 14 teacher-only; 1 learner-only; 20 both lost. Learned 66/100 vs teacher
79/100; gap -13 percentage points, paired SE 3.67 points. Approximate 95% normal
interval -20.2 to -5.8 points; monitoring comparison is descriptive, not untouched test.

## Strategy findings

On the 14 teacher-only cases, learner played Demon Form in 13/14 and Corruption in
14/14. Teacher played both in 14/14. This is not a wholesale failure to discover the
engine. Conditional mean first Demon Form turn: learner 2.38 vs teacher 2.50 (zero-based).
These means do not establish correct timing.

The clearest descriptive strategic difference is Dual Wield selection:

| Selected card | Teacher, all 100 fights | Learner, all 100 fights | Teacher, 14 teacher-only fights | Learner, same 14 |
|---|---:|---:|---:|---:|
| Clash+ | 51 | 46 | 9 | 4 |
| Anger+ | 38 | 3 | 6 | 0 |
| Sword Boomerang+ | 5 | 46 | 1 | 8 |
| Bite | 10 | 3 | 0 | 1 |
| Demon Form+ | 1 | 0 | 0 | 0 |
| Bash+ | 0 | 1 | 0 | 0 |

Counts are selection decisions, not independent fights, and depend on what cards were
available. Teacher sometimes plays Dual Wield without a following selection; only
observed selection decisions counted. These are associations, not proof of optimality.

Plausible interpretation: learner over-favours copying Boomerang relative to zero-cost
attacks. With three energy, strength applies to each zero-cost attack too; Corruption
can enable Clash; Pyramid makes redundant one-cost attacks potentially costly to hold.
The initial human hypothesis 'avoid Anger' is not supported as a blanket rule here:
teacher plays more Anger, including on teacher-only wins (mean 4.64 vs learner 1.93 plays).
Longer fights and different opportunities confound that mean.

Examples, existing monitor IDs:
- 23: identical physical setup through first two turns (first bit divergence only Bash/Defend
  ordering). Teacher plays Corruption on turn 2; learner spends energy on Cleave/Bite and
  delays Corruption to turn 3, then ends turn with recorded search value ~96 and later dies.
  Teacher later copies Clash; learner copies Boomerang. No move-rescue counterfactual run.
- 59: learner never plays Demon Form and dies; teacher establishes it on turn 3 and wins.
- 83: teacher opens Spot Weakness then Headbutt/Bite; learner Headbutt/Bash, loses initial
  strength and later copies Boomerang while teacher copies Clash.
- 91: teacher copies Clash on the Demon Form turn; learner plays only Demon Form then
  copies Boomerang next turn. These observations do not uniquely isolate causes.

## Value / policy

Raw NN value at initial state: mean 63.26/100 vs actual learner 66/100 (n=100).
Search root value: mean 71.33/100. Aggregate values are not collapsed; however aggregate
calibration does not establish good action ranking or later-state calibration.

First bit divergence on 14 teacher-only fights: NN prior favours learner action over
teacher action in 11 cases, ties for two identical-card selections/orderings, favours
teacher in one. First divergences can be cosmetic; none are established causal mistakes.

Example 83, identical starting state: Spot Weakness prior 13.0%, Headbutt 40.8%; learner
search assigns Spot Weakness 36 visits / value 33.0, Headbutt 1,820 / value 52.7. The
successful teacher alternative is legal but underexplored and valued poorly by this
search. Budget, prior and evaluator contributions remain confounded.

## Recommendation

Do not simply extend unchanged self-play. First consider a bounded teacher-replay / teacher
policy refresh branch (no architecture changes), comparing to saved update15 on the same
monitor seeds. Teacher-labelled learner states remain a possible intervention, but require
new annotation machinery and outcome-safe value labels. Before implementing that larger
change, the zero-cost attack vs Boomerang-copy pattern provides a concrete diagnostic.
No causal claim or hard-coded strategy rule is warranted yet.

Artifacts: run `out/gap-investigation/{reference-monitor,iter015-eval}/rows.parquet`,
`out/gap-investigation/analysis.json`. Reproduction script for core statistics:

```sh
PYTHONPATH=. .venv/bin/python experiments/single-deck-expert-iteration/gap_analysis.py --out runs/schema=combat_v4/date=2026-10-06/id=single-deck-demon-form-v1/out
```
