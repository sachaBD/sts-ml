# First bounded run — completed

The subsequent untouched 600-seed confirmation is recorded in [FINAL_RESULTS.md](FINAL_RESULTS.md).
The final-test-not-run statements below describe the state immediately after training.

Fixed Barricade/Entrench/Body Slam deck, 41/75 HP, fixed relics/counters, no potions.
Fresh random width64 network (no D5 initialization), bounded sigmoid value head.
200 MCTS20k bootstrap training fights, then five updates of 100 fresh learner2k fights.
Total run time: 30 minutes, 8 workers. All scheduled stages completed.

| monitoring checkpoint | training fights generated | wins / 100 | 95% Wilson interval | paired gap to MCTS ±1 SE |
|---|---:|---:|---:|---:|
| After teacher bootstrap | 200 | 94 | 87.5–97.2% | +4.0 ±2.4 points |
| After five learner updates | 700 | **99** | **94.6–99.8%** | **+9.0 ±2.9 points** |

MCTS20k reference: 90/100 wins on the same fixed monitoring starts. Final model learner2k
aggregate searched-decision time: 745 seconds versus MCTS reference 1,189 seconds across
100 fights each; timing variability has not been quantified. Different simulation budgets
and teacher HP-sensitive versus learned win-only objectives remain material differences.

Exploratory collection wins by update: 91, 95, 97, 99, 98 out of 100 fresh fights each.
These are different seeds with exploration enabled, not the fixed evaluation curve.
At update five replay still included 120 teacher + 500 learner fights, so this is not
self-only learning yet. Scheduled taper reaches zero teacher fights at update eleven.

Interpretation: strong evidence the pipeline can specialize from scratch on this task,
not proof of general superiority or 100% theoretical winnability. Monitoring checkpoint
selection is adaptive; its intervals/gap are descriptive. **The reserved 600-seed final
test has not been played.** Recommended next action is fresh confirmation before making
a stronger performance claim or deciding whether to extend to self-only training.

Run: `runs/schema=combat_v4/date=2026-10-05/id=single-deck-fresh-v1/`.
Plot: run `out/curve.png`; detailed report: `out/REPORT.md`; raw curve: `out/curve.csv`.
Latest model: run `out/iter005/model/`.

Read-only overall dashboard:

```sh
watch -n 5 '.venv/bin/python -m apps.run_rl.single_deck_status --run runs/schema=combat_v4/date=2026-10-05/id=single-deck-fresh-v1'
```
