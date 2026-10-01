# Easy-pool fights: how much HP does cheaper search cost? (2026-09-29)

**Why:** macro research (card picking) needs to play huge numbers of fights, so it needs a cheap combat agent. Easy-pool
fights (Cultist, Jaw Worm, Small Slimes, Two Louse, i.e. a run's first 3 regular fights) are never lost; what
matters is **the HP they cost**.

**Question:** how much extra HP per fight does each cheaper agent lose compared with MCTS at 20k simulations?

## Answer

**Plain MCTS (guided rollouts, 8 particles) at 100–200 simulations per decision.** It loses 0.3–0.4 HP per fight more
than MCTS 20k and is 80–160× cheaper. Nothing tested beats plain MCTS at a matched budget.

| MCTS simulations | HP lost per fight | extra HP vs 20k [95% CI] | same, on fresh fights | extra HP per run (×3 fights) | search time per fight |
|---:|---:|---:|---:|---:|---:|
| 10 | 7.67 | +2.15 [+1.91, +2.40] | | +6.5 | 1.6 ms |
| 25 | 6.75 | +1.23 [+1.05, +1.41] | | +3.7 | 3.2 ms |
| 50 | 6.26 | +0.74 [+0.59, +0.90] | | +2.2 | 5.8 ms |
| **100** | **5.95** | **+0.43** [+0.30, +0.58] | +0.40 [+0.25, +0.55] | **+1.3** | **11 ms** |
| **200** | **5.79** | **+0.27** [+0.14, +0.41] | +0.33 [+0.20, +0.47] | **+0.8** | **22 ms** |
| 500 (current eval setting) | 5.73 | +0.22 [+0.10, +0.34] | +0.20 [+0.07, +0.32] | +0.7 | 64 ms |
| 1k | 5.71 | +0.19 [+0.08, +0.31] | | +0.6 | 122 ms |
| 5k | 5.62 | +0.10 [+0.01, +0.20] | | +0.3 | 554 ms |
| 20k (reference) | 5.52 | — | — | — | 1,750 ms |

Dev fights: n = 983. Fresh fights: n = 985, a second fight set untouched until the end. Search time is for one CPU
thread; 11 ms ≈ 90 fights per CPU-second.

## What to remember

1. **HP cost rises smoothly as the budget falls; there's no cliff.** Above ~200 simulations extra compute buys very
   little (Fig 1, Fig 3). Below 100 every halving of the budget roughly doubles the extra HP lost.
2. **The results replicate.** The three budgets re-run on fresh fights landed within their CIs (+0.43 → +0.40,
   +0.27 → +0.33, +0.22 → +0.20).
3. **Jaw Worm is the expensive fight** (8.3 HP even at 20k) and is saturated from 200 simulations. **Two Louse barely
   depends on search** (Fig 2).
4. **Cheap search mostly adds bad outcomes**: fewer 0-HP fights and a longer tail of expensive ones (Fig 5). At 100
   simulations, 15% of fights cost > 2 HP more than 20k, against 6% for 20k vs itself (Fig 6).
5. **The alternatives are all worse** at the same budget (Fig 4):
   - Fewer particles lose HP and save no time.
   - The gen1 value net is +1.6 at 200 and +0.3 at 1k.
   - The elite-trained t5 policy net is about +5: it's out of distribution here.
6. **Later easy fights look like floor 1.** The 2nd and 3rd easy fights (decks built from Neow and card picks, lower
   HP) show the same picture. Low budgets lose slightly more there (fresh fights, 100 simulations, HP-equivalent: −0.2
   on the 1st fight, −0.5 to −0.6 on the 2nd/3rd; `results/hpeq/`).
7. **Losses:** 0 in about 39,000 MCTS fight plays. The only deaths were in the weakest net arms (gen1 at 50, t5 at 10).
8. **For a macro harness, search is no longer the bottleneck below ~200 simulations.** This replay harness spends
   ~30 ms per fight outside the search (process start, run rebuild, row recording), against 11 ms of search at 100.
   Running fights inside one process would gain more than cutting simulations further.

## Figures (`results/`)

| | |
|---|---|
| ![](results/fig1_extra_hp_vs_budget.png) | **Fig 1:** extra HP lost vs budget, dev and fresh fights. The headline. |
| ![](results/fig2_hp_lost_by_encounter.png) | **Fig 2:** HP lost per fight by encounter. |
| ![](results/fig3_cost_vs_hp.png) | **Fig 3:** the trade-off: extra HP vs search time (and fights per CPU-second). |
| ![](results/fig4_alternatives.png) | **Fig 4:** fewer particles and value/policy nets vs plain MCTS at matched budgets. |
| ![](results/fig5_hp_lost_distribution.png) | **Fig 5:** distribution of HP lost per fight (10 vs 100 vs 20k). |
| ![](results/fig6_worse_or_better.png) | **Fig 6:** how often one play is > 2 HP worse or better than 20k on the same fight, with the 20k-vs-itself noise. |

Full tables with every arm and every encounter: `results/hp_lost_dev.md` and `results/hp_lost_confirm.md`.

## How it was measured

- **Fights:** 250 per encounter, covering all three easy fights of a run, taken from stored A20 Ironclad runs
  (`act1-all-bosses-a20-scaled-search`). They were chosen by hash, never by outcome, and frozen in advance
  (`make_fights.py`).
  - Dev set: run buckets 0–1 (`fights.csv`). Fresh set: buckets 2–3 (`confirm.csv`, played once at the end).
  - 15–17 fights per set were skipped because the current simulator no longer replays them identically.
- **Paired design:** every agent replays the same fight starts (same deck, HP and game RNG). Δ is a per-fight
  difference, which removes deck-to-deck variance.
- **Search seeds:** each agent played every fight with 2 search seeds (20k with 4), and each fight is scored by its
  mean. As a sanity check, 20k vs itself with different seeds differs by +0.03 [−0.05, +0.11] HP, i.e. nothing.
- **HP lost** = starting HP − final HP, after Burning Blood's +6, so it can be negative.
  - Mean = the 4 encounters weighted equally.
  - CIs = 95% cluster bootstrap over the source run seeds (fights from one run share a deck).
- **Search time** is measured inside the worker (move choosing only) on a Ryzen 3600X (6 cores / 12 threads), with 11
  workers running at once, so it is per-thread under full load.
  - Particle-arm timings are not comparable: most ran while another job shared the CPU, which doubled their times.
    Re-timed on an idle machine, 1/2/4 particles cost the same as 8.
  - The MCTS sweep and net arms ran on an otherwise idle machine.
- **Potions:** cheaper agents drink slightly fewer potions. If a kept potion counts as 4 HP, their gaps shrink a
  little (e.g. MCTS 500: +0.22 HP, +0.13 "HP-equivalent"). The HP-equivalent analysis is in `results/hpeq/`.

## Not covered

- Hard-pool fights, elites and bosses. They need more search (elite-bench found 20k ≫ 1k), so they need their own
  sweep.
- Whole-run impact of +1 HP per run from easy fights (probably small, since rest sites heal).

## Files

| file | purpose |
|---|---|
| `PLAN.md` | pre-registered plan (goal: ≤ 0.5 extra HP per fight) |
| `make_fights.py` | writes `fights.csv` / `confirm.csv` |
| `make_configs.py` | writes `configs/*.toml` (one run per agent and seed) |
| `run.sh` | runs configs one at a time (`run.sh ID...`); logs in `logs/` |
| `report.py` | this README's tables and figures |
| `analyze.py` | the older HP-equivalent analysis (`results/hpeq/`) |

- Runs: `runs/schema=combat_v3/date=2026-09-29/id=easy-cheap-*` (dev), `id=easy-cheap-confirm-*` (fresh),
  `id=easy-retime-*` (timing re-checks).
- Code change for timing: `agents/teacher_search.{hpp,cpp}` (`search_seconds()`), and `apps/value_play/worker.cpp` and
  `play.py` (per-fight `search_seconds` in `summary.json`).
- Reproduce the report: `PYTHONPATH=python .venv/bin/python
  experiments/act-1-combat/2026-09-29-easy-cheap-agent/report.py`. This needs `matplotlib`, which was pip-installed into
  `.venv` and isn't in requirements.txt.
