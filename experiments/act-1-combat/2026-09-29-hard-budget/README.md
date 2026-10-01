# Hard-pool fights: HP lost by MCTS budget (2026-09-29)

**Why:** macro research (card picking) needs cheap combat. Hard-pool fights are every regular fight after a run's first
3 (10 encounters, ~1.6 per act-1 run). Unlike easy fights, they can kill you.

**Question:** how much HP, and how many deaths, does each MCTS budget from 500 to 10k cost? This is a comparison
between budgets. 10k is simply the most expensive one run, not a claimed optimum.

## Answer

**Every step up in budget saves a little HP, with no plateau yet at 10k. Deaths don't depend on budget in this
range.**

- Going from 500 to 10k saves 0.5 HP per fight (~0.8 HP per run).
- The current harness setting, **2k**, is 0.25 HP per fight behind 10k at a fifth of the cost.

| MCTS sims | HP lost / fight | extra HP vs 10k [95% CI] | vs next budget up | per run (× 1.6) | deaths | search time / fight |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 7.78 | +0.52 [+0.36, +0.68] | +0.07 [−0.07, +0.21] | +0.83 | 1.0% (19) | 95 ms |
| 1k | 7.71 | +0.45 [+0.28, +0.62] | +0.20 [+0.08, +0.33] | +0.72 | 1.1% (21) | 168 ms |
| **2k** (harness) | 7.51 | +0.25 [+0.12, +0.38] | +0.13 [+0.00, +0.26] | +0.40 | 1.1% (20) | 308 ms |
| 5k | 7.38 | +0.12 [+0.02, +0.22] | +0.12 [+0.02, +0.22] | +0.19 | 1.0% (17) | 750 ms |
| 10k | 7.26 | — | — | — | 0.9% (17) | 1,422 ms |

- **Fights:** n = 889 fights × 2 search seeds per budget.
- **HP lost** counts a death as losing all starting HP.
- **Pooled:** encounters are weighted by how often they occur in real runs.
- **Noise check:** 10k against itself (seed 1 minus seed 0) = −0.08 [−0.25, +0.09] HP per fight.

## What to remember

1. **Hard fights need about 10× the budget easy fights need for the same shortfall.** 2k here is about as far behind
   the top as 200 was on easy fights (+0.25 vs +0.27). HP per fight falls steadily to 10k, roughly 0.1–0.2 HP per
   doubling of the budget.
2. **Deaths are the same fights at every budget.** 16 fights had any death; 7 of them died in 8–10 of their 10 plays
   across budgets, so they're probably lost before the fight starts. Death rate is flat at ~1% from 500 to 10k.
3. **Gremlin Gang (13.8–15.1 HP, most deaths) and Exordium Thugs (~11 HP) are the expensive fights** (Fig 2).
   Exordium Wildlife, Three Louse and Gremlin Gang gain the most from search. Two Fungi Beasts, Blue Slaver and
   Red Slaver barely change.
4. **Cost:** search time roughly doubles with each budget step. Search is the bottleneck here, not the harness
   (0.1–1.4 s per fight vs ~30 ms of overhead).

## Figures (`results/`)

| | |
|---|---|
| ![](results/fig1_hp_and_deaths_vs_budget.png) | **Fig 1:** extra HP lost vs 10k (paired) and death rate, by budget. The headline. |
| ![](results/fig2_by_encounter.png) | **Fig 2:** HP lost per fight for each encounter, with deaths marked. |
| ![](results/fig3_cost_vs_hp.png) | **Fig 3:** HP lost vs search time (and fights per CPU-second). |
| ![](results/fig4_worse_or_better.png) | **Fig 4:** how often one play is > 2 HP worse or better than 10k on the same fight, with the 10k-vs-itself noise. |

Full tables, including per encounter: `results/tables.md`.

## How it was measured

- **Fights:** 100 per encounter from stored A20 Ironclad runs (`act1-all-bosses-a20-scaled-search`, buckets 0–1),
  chosen by hash and frozen in `fights.csv`. 111 of the 1,000 no longer replay identically on the current simulator
  and were skipped (the same ones for every arm).
  - `confirm.csv` (buckets 2–3) is written but unused.
- **Arms:** MCTS (guided rollouts, 8 particles, fair play) at 500 / 1k / 2k / 5k / 10k, each with search seeds 0 and
  1. Every arm replays the same fight starts (deck, HP, game RNG), so differences are per fight.
- **CIs:** 95% cluster bootstrap over source run seeds.
- **Search time:** measured inside the worker, one thread, 11 workers on a Ryzen 3600X. The machine had no other jobs
  (load is logged per run in `logs/main.log`).
- **Precision:** pooled paired differences are ±0.1–0.17 HP. Per encounter there are only ~100 fights, so the
  encounter numbers flag where to look rather than ranking close ones. With ~20 deaths per arm, only large
  death-rate differences would show.
- **Wall time:** pilot 1 min; main run 16 min (20:16–20:32Z).

## Not covered

- Budgets above 10k (HP was still falling at 10k).
- Fresh-fight confirmation.
- Elites and bosses.

## Files

| file | purpose |
|---|---|
| `PLAN.md` | draft design; the agreed changes are recorded in this README |
| `make_fights.py` | writes `fights.csv` / `confirm.csv` |
| `make_configs.py` | writes `configs/*.toml` |
| `run.sh` | runs configs one at a time (`run.sh ID...`); logs in `logs/` |
| `report.py` | this README's tables and figures |

Runs: `runs/schema=combat_v3/date=2026-09-29/id=hard-budget-mcts-*`.
