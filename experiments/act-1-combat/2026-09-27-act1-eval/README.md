# Full act 1: MCTS teacher vs value net ab-gen1 (2026-09-27)

Same 1,000 seeds (600000000000..+999), A20, all three bosses, 8 particles, no random moves, no search tweaks, and
simulations easy 500 / hard 2000 / elite 5000 / event 5000 / boss 15000. SimpleAgent plays outside combat. The only
difference between the runs is the search leaf.

| run | config | leaf |
|---|---|---|
| A `combat_v3/2026-09-27/act1-eval-mcts-a20` | `apps/bootstrap/config/act1-eval-mcts.toml` | guided_rollout |
| B `combat_v3/2026-09-27/act1-eval-ab-gen1-a20` | `apps/bootstrap/config/act1-eval-ab-gen1.toml` | value_net (`value_net_v1/2026-09-27/ab-gen1`) |

Reproduce the table: `PYTHONPATH=. .venv/bin/python experiments/act-1-combat/2026-09-27-act1-eval/compare.py`
Per-seed results: `select * from act1_results where run_id like '%act1-eval-%'` (`runs.query.connect()`).

## Result (n = 1,000 paired seeds; ± = 1 standard error)

| | MCTS | ab-gen1 | ab-gen1 − MCTS |
|---|---:|---:|---:|
| act cleared | 60.6% ±1.5 | 56.3% ±1.6 | **−4.3 pts ±1.5** |
| reached boss | 89.1% ±1.0 | 84.8% ±1.1 | −4.3 pts ±1.0 |
| final HP (0 if died) | 19.7 | 17.8 | −1.9 ±0.5 |
| cleared, Slime Boss seeds (337) | 63.5% | 65.3% | +1.8 pts ±2.2 |
| cleared, Guardian seeds (350) | 66.0% | 56.9% | −9.1 pts ±2.6 |
| cleared, Hexaghost seeds (313) | 51.4% | 46.0% | −5.4 pts ±2.9 |

Paired clears: both 474, MCTS only 132, net only 89, neither 305.
Deaths before the boss are mostly at elites: Three Sentries 45 → 67, Lagavulin 28 → 47, Gremlin Nob 11 → 18.

## By fight category (`fights.py`)

All fights played (each agent's own runs; later fights differ between the runs). Cells are MCTS / ab-gen1;
± = 1 SE.

| category | fights | deaths | death rate | net − MCTS (pts) | mean start HP |
|---|---:|---:|---:|---:|---:|
| easy | 2940 / 2921 | 2 / 1 | 0.1% / 0.0% | −0.0 ±0.1 | 63.3 / 62.7 |
| hard | 1636 / 1595 | 9 / 6 | 0.6% / 0.4% | −0.2 ±0.2 | 54.0 / 52.8 |
| event | 135 / 132 | 14 / 13 | 10.4% / 9.8% | −0.5 ±3.7 | 50.3 / 48.9 |
| elite | 1286 / 1260 | 83 / 128 | 6.5% / 10.2% | **+3.7 ±1.1** | 56.9 / 55.6 |
| boss | 891 / 848 | 285 / 285 | 32.0% / 33.6% | +1.6 ±2.3 | 54.3 / 53.6 |

Deaths outside combat (events): MCTS 1, ab-gen1 4.

Matched fights (identical start in both runs; paired; HP loss counts a death as its starting HP):

| category | fights | deaths | HP loss | net − MCTS |
|---|---:|---:|---:|---:|
| easy | 1768 | 0 / 0 | 6.64 / 7.44 | **+0.80 ±0.09** |
| hard | 286 | 1 / 1 | 8.41 / 8.26 | −0.15 ±0.30 |
| event | 14 | 0 / 0 | 15.07 / 16.71 | +1.64 ±1.78 |
| elite | 294 | 17 / 25 | 29.86 / 31.55 | **+1.69 ±0.48** |
| boss | 113 | 35 / 40 | 37.33 / 39.60 | +2.27 ±1.04 |
