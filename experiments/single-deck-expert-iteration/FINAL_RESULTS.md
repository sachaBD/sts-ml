# Final confirmation: learned search beats the rollout teacher on this deck

Preselected update-5 specialist; **600 previously untouched seed-derived encounters**.
Fixed Barricade/Entrench/Body Slam deck, 41/75 HP, fixed relics/counters, no potions.
Same starts and frozen native worker for both agents. No retraining, target generation for
learning, or checkpoint selection on these final seeds. All 1,200 plays completed.

| Agent | Wins | Win rate | 95% Wilson interval |
|---|---:|---:|---:|
| Fresh learned network + search, 2k sims | **595/600** | **99.17%** | **98.06–99.64%** |
| Guided-rollout MCTS, 20k sims | 514/600 | 85.67% | 82.64–88.24% |

**Paired advantage: +13.50 percentage points.** Paired SE: 1.42 points; approximate
95% normal interval: **+10.72 to +16.28 points** across the 600 independent seed trials.

Paired outcomes:
- Both won: 513.
- Learned won, MCTS lost: **82**.
- MCTS won, learned lost: **1**.
- Both lost: 4.

Measured searched-decision time per fight (n=600 each): learned mean 7.93 seconds,
median 7.80; MCTS mean 12.06 seconds, median 12.05. Aggregate learned search time was
about 34% lower. These are observed timing summaries, not uncertainty intervals; timing
variability/system-load uncertainty has not been quantified. They exclude encoding,
training and other orchestration overhead.

## What this establishes

The current architecture can learn a strong specialist from random initialization,
using 200 teacher bootstrap fights + 500 learner fights and the stated curriculum.
It can outperform the deployed MCTS20k baseline on this fixed task, at less measured
search time. This is not merely a training-loss improvement or the earlier 100-seed
monitoring result. D5 was not used to initialize the model.

## What it does not establish

- General superiority across decks/fights, 100% theoretical winnability, or a formal
  upper bound. Only one fixed loadout/HP condition was tested.
- Pure self-learning success: update five still sampled 120 teacher replay fights alongside
  500 learner fights. Teacher taper reaches zero at update eleven, which has not run.
- That any individual design change caused the gain. Architecture head, task concentration,
  teacher bootstrap and replay/learning changes were bundled, not separately ablated.
- Equal search objectives: learned search uses win-only terminal values, while rollout
  teacher search also values HP/resources. The result compares deployed recipes; some
  advantage may be objective alignment rather than learned evaluation alone.

## Next recommendation

Add one contrasting deck (the queued Flex/JAX/Limit Break/Heavy Blade deck is a candidate),
bootstrap its strategy and train jointly while retaining this deck's replay. Track both
separately to expose forgetting. These 600 seeds are now a consumed final test; if used
for future regression monitoring they cannot also remain an untouched final set.

## Artifacts

- Training: `runs/schema=combat_v4/date=2026-10-05/id=single-deck-fresh-v1/`.
- Final confirmation: `runs/schema=combat_v4/date=2026-10-05/id=single-deck-fresh-final/`.
- Detailed final report: final run `out/REPORT.md`, `out/summary.json`.
- Model: training run `out/iter005/model/`; final frozen copy: final run `out/frozen/model/`.
- Exact actions/search annotations: final run `out/{learned,mcts}/`.
- Learning curve: training run `out/curve.png`, `curve.csv`, `curve.json`.

Dashboard (both stages now DONE):

```sh
.venv/bin/python -m apps.run_rl.single_deck_status --run runs/schema=combat_v4/date=2026-10-05/id=single-deck-fresh-v1
```
