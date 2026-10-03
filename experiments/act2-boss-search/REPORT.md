# Act 2 bosses: does 100k-simulation MCTS beat 20k? (2026-10-02)

## Answer

**No meaningful gain.** On the same 300 Act 2 boss fights (100 per boss), 100k simulations won 37.3% vs 35.0% for 20k:
**+2.3 pp, 95% CI (−2.0, +6.7)**, McNemar p = 0.37. A gain larger than about 7 pp is ruled out. 100k costs 4.8× the
time per fight (50 s vs 10 s).

| Boss | n | 20k win | 100k win | 100k − 20k (95% CI) | fights won only by 20k / only by 100k |
|---|---:|---:|---:|---|---:|
| **All** | 300 | 35.0% | 37.3% | **+2.3 pp (−2.0, +6.7)** | 19 / 26 |
| Automaton | 100 | 31.0% | 33.0% | +2.0 pp (−6.3, +10.3) | 8 / 10 |
| Champ | 100 | 34.0% | 31.0% | −3.0 pp (−8.9, +2.9) | 6 / 3 |
| Collector | 100 | 40.0% | 48.0% | +8.0 pp (−0.2, +16.2) | 5 / 13 |

The pooled row is the test. Per-boss rows are descriptive: three subgroups, n = 100 each, no multiple-comparison
correction. Collector is the only one that leans positive, and it is not significant even before correction.

## What it says about the fights

- **Most outcomes don't depend on the budget.** 255 of 300 fights (85%) had the same result at 20k and 100k.
- **Losses are not close.** Boss HP left when we die averages **47%** (20k) and **48%** (100k). Champ 37–38%,
  Automaton 50–51%, Collector 54–60%. The search isn't narrowly missing wins; these decks/HP lose by a lot under
  this search.
- **Fight style barely changes with budget:** same length (6.4 vs 6.5 turns), same HP left on wins (27.0 vs 27.3),
  same potion use (79% vs 77% of held potions drunk).
- **Side note:** 31 of 195 losses at 20k (16%) died still holding a potion (35 / 188 at 100k). Not investigated;
  some may be potions that couldn't have helped.

## Setup

- **Fights:** Act 2 boss fights from `act2-fresh-final` (selected overworld model `act2-r02-train`, greedy, rollout
  combat), 100 sampled per boss (sample seed 0, out of 153 / 149 / 138).
- **Rebuild** (`apps/boss_rebuild`): new game from the original run seed; recorded deck (upgrades, misc), relics
  (with counters, added without pickup effects), potions, HP and max HP loaded; Act 2 boss room at floor 33. Battle-start
  HP set to the original fight's (covers Pantograph etc.). Not reproduced: which card a bottled relic holds (7% of
  runs carry one), and miscellaneous/potion random streams.
- **Rebuild is faithful:** rebuilt 20k won 35.0% vs 35.7% in the original fights (−0.7 ± 1.4 pp, 1 SE);
  outcomes matched on 282 of 300 fights.
- **Play:** guided-rollout MCTS, 8 particles, same as in the runs; only the simulation cap differs. Each arm plays
  every fight once. 10 workers, 31 min total.
- **Statistics:** paired by fight; difference ± 1.96 SE of the paired difference; exact McNemar test on discordant fights.

## Implications

- Raising the boss budget is not worth it: at most a few pp of boss wins (≈ +1 pp Act 2 clears) for ~+40 s per run.
- This tests only the budget of the existing search, not its rollout policy or leaf evaluation. A different
  evaluator (e.g. a value net) could still change which plans are found. But with losses leaving the boss at about
  half HP, how many of these fights are winnable at all is the open question.
- **Cheap next check (suggested):** play the 195 lost fights with the perfect-information oracle search (true RNG,
  max backup). The share it wins bounds what any combat improvement can recover; the rest has to come from the deck
  and HP entering the fight (overworld).

## Artifacts

- Run: `runs/schema=boss_bench_v1/date=2026-10-02/id=act2-boss-20k-vs-100k` (`fights.jsonl`, `results.jsonl`).
- Launch: `launch.sh`; analysis: `analyze.py` → `analysis.log`.
- Code: `apps/boss_rebuild/worker.cpp` (CMake target `boss_rebuild_worker`), `apps/boss_rebuild/bench.py`.
