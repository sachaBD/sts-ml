# combat_v4: recording test and write / read performance (2026-10-02)

## Result

The new combat_v4 (`runs/schema=combat_v4/schema.py`: a `fights` table for replay, plus an optional `search`
table) records and rebuilds cleanly, and storage cost is negligible next to search:

- **2,992 fights, 35,306 decisions: every fight was rebuilt from its stored start + actions with sts_lightspeed only and
  matched the played fight's complete final battle state.** A separate reader (`combat_v4_replay`) rebuilt them
  all again from the parquet file.
- **Size:** 167 KB of replay facts (56 B per fight, 4.7 B per decision) + 1.19 MB of search statistics (41 B per
  searched decision). Search statistics are 88% of the total.
- **Writing costs about 1.6% of the worker's time, and the check rebuild is most of that.** Encoding is 0.6%;
  Python/Arrow/parquet adds about 0.3 s per 3,000 fights.
- **Reading is fast:** both tables load in under 5 ms each; DuckDB queries, including the fights x search join, take
  2–13 ms; rebuilding every fight takes 0.04 s (about 0.9 M decisions/s on one thread).

## Setup

- 1,000 seeded Ironclad A20 act 1 runs (seeds 990000000000+), first 3 fights each (8 runs died sooner), SimpleAgent
  out of combat, guided-rollout MCTS with 500 simulations per decision and 8 particles, forced moves unsearched.
  Mostly floors 1–5; 2,942 of 2,992 fights won. Same seeds recorded twice: search stats `on` and `off`.
- Runs: `runs/schema=combat_v4/date=2026-10-02/id=act1-early-stats-on` and `...-stats-off`.
- Code: `environments/combat/record_v4.{hpp,cpp}` (start record + rebuild, sts_lightspeed only),
  `apps/combat_record/worker.cpp` (play, record, check), `apps/combat_record/record.py` (parallel run + parquet),
  `apps/combat_record/replay.cpp` (independent rebuild + check), `bench_read.py` (read timings, `bench_read.log`).
- 10 worker processes, 12-core machine, otherwise idle. Timings are single measurements for writing; reads are
  the median of 5 repeats (min–max in `bench_read.log`). No variability estimate for write timings.

## Storage

| Table | Rows | File | Per row | Per decision |
|---|---:|---:|---:|---:|
| fights (replay facts) | 2,992 | 167 KB | 56 B | 4.7 B |
| search (stats on) | 29,024 | 1,194 KB | 41 B | 33.8 B |
| **Total, stats on** | | **1.36 MB** | | **38.6 B** |
| **Total, stats off** | | **167 KB** | | **4.7 B** |

Search stats `on` and `off` give the same fights file (167,096 vs 167,068 B; only fight_id prefixes differ). These are
early-act decks, mostly starter cards: later-act decks will make start records larger (not measured).

## Writing (stats on; worker seconds summed over 10 processes)

| Stage | Seconds | Share |
|---|---:|---:|
| MCTS search | 120.3 | 98.4% |
| Recording: start record, JSON building and printing | 0.74 | 0.6% |
| Check: rebuild every fight + compare full final state | 1.29 | 1.1% |
| Python: parse worker JSON (16 MB) | 0.23 | wall, after play |
| Arrow build (fights + search) | 0.09 | wall |
| Parquet write, zstd (fights + search) | 0.03 | wall |

Stats off: recording 0.20 s, JSON parse 0.05 s (3.5 MB). Wall time to play was 12.6 s either way. The search budget
here is the cheapest one used (500 simulations); at 5k–100k the search share only grows.

## Reading (stats-on run, median of 5)

| Operation | Seconds |
|---|---:|
| pyarrow: read fights / search table | 0.004 / 0.005 |
| pyarrow → Python dicts: fights / search | 0.30 / 0.80 |
| DuckDB: win rate by encounter | 0.002 |
| DuckDB: unnest every search child (≈ 130k) | 0.006 |
| DuckDB: join fights x search, visit share of the move played | 0.013 |
| C++ rebuild + check all 2,992 fights (one process, one thread) | 0.10 total: 0.04 JSON parse + 0.04 rebuild |

- Conversion to Python objects is the only slow read path (0.8 s for 29k search rows); DuckDB / Arrow avoid it.
- Join check: all 29,024 search rows join to a fight, and every one contains the move actually played
  (mean visit share of the played move 0.33).

## Notes

- The search budget dominates writing completely; recording format choices don't change collection speed.
- Not done: the combat_v4_full decompressor (per-decision public / hidden state), batching for long collections
  (here all fights go to one file at the end), and switching the run_rl writer / boss bench to combat_v4. Old
  readers of the previous v4 layout (`apps/run_rl/records.py`, `apps/run_rl/train_combat.py`,
  `agents/combat/value/data.py`) are still broken.
