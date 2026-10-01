# Teacher search performance (2026-09-29)

This covers speedups to the teacher, the MCTS that plays our fights. It builds on `search_perf.md`. The motivating
workload is `apps/card_marginals`, which plays thousands of fights and keeps only their outcomes.

## Background

- **What the teacher is.** At every decision of a fight it runs a Monte Carlo tree search
  (`PublicBeliefCombatSearch` in sts_lightspeed, driven by `agents/combat/search/teacher_*.cpp`).
- **Simulations.** Each simulation walks down the tree, then plays the fight to the end with a cheap heuristic
  policy (a "guided rollout"). The number of simulations per decision is the budget.
- **Budgets.** Easy fights get 500, hard 2k, elite 5k, and boss 15k.
- **Particles.** The search averages over 8 "particles", which are guesses of the hidden draw order.
- **Where the time goes.** Boss fights are 3–10 s each on one core and dominate the cost. Easy fights are about 0.05 s.

## Where the time goes

This profile is from a boss fight on a Release build (`-O3`, before the changes below). It was measured with a
home-made sampling profiler (`scratch/profile_teacher/`; the machine has no `perf`), with about 4,900 samples.

| Part | Share of CPU |
|---|---|
| Rollouts (playing each simulation to the end) | 64–68% |
| &nbsp;&nbsp;game engine executing actions | 31% |
| &nbsp;&nbsp;listing the legal actions | 17% |
| &nbsp;&nbsp;heuristic picking the move (`SimpleAgent`) | 14% |
| Tree walk: engine steps, state hashing (`observationKey`, 7%), node lookup and selection | ~31% |
| Everything outside the search (row recording, JSON, setup) | <1% for bosses, ~5% for easy fights |

## What was changed

"Identical" means that, for the same inputs, the fights play the same moves and give the same outcomes.

| Change | Where | Effect (CPU time, new / old) | Identical? |
|---|---|---|---|
| **LTO** (link-time optimization). Lets the compiler inline sts_lightspeed's many tiny getters across files. On by default. | `CMakeLists.txt` | 0.67–0.69 | Yes (166/166 fights) |
| **Lazy rollout.** The rollout listed every legal action even though 80% of the time it plays the heuristic's move. Now it lists them only when it needs the fallback random move. The random draws are unchanged. | sts_lightspeed `BattleScumSearcher2::rolloutAction` | ~0.87 (boss and elite) | Yes (172/172) |
| **No-record fight** (`play_fight` overload without rows). It skips state encoding and row building, and plays a forced move (only one legal move) without searching it. | `agents/combat/search/teacher_search.cpp`; used by card_marginals | Easy fights 0.65 together with the lazy rollout; small on boss fights | Yes |
| **Tree reuse with a top-up budget** (teacher setting `tree_reuse = true`) | `agents/combat/search/teacher_leaves.cpp` | ~0.7 of search time on boss fights (see below) | No, the policy changes slightly |

The first three together: boss fights ~0.61× CPU (1.65× throughput), easy fights ~0.44× (2.25×).

**How the identical changes were measured.** Worker requests from the real card_marginals configs were run on one
core, old and new binaries interleaved, 2 rounds each: 6 boss, 6 elite and 160 easy fights. The runs agreed within
about 3%. Fight results were compared field for field.

## Tree reuse

- **What it does.** After a move is played, the part of the tree under that move is kept and becomes the next
  search's starting point. The visits it already holds count toward the budget: the next search runs
  `max(N/10, N − kept visits)` new simulations.
- **Before this change.** Reuse existed (`play_fight` `reuse`) but was never switched on. Kept visits were extra
  evidence on top of a full budget, so it saved almost no compute.
- **Measurement.** Slime Boss at 15k simulations, one fight of 32 decisions, so treat the numbers as indicative:
  - The kept tree held on average 51% of a fresh search's simulations.
  - At the start of a turn it holds none, because the newly drawn hand never matches one of the 8 guessed draw orders.
  - Mid-turn it holds 4–11k.
  - About 70% of decisions are mid-turn (8,888 fights of the act1 eval cohort).
  - The top-up budget therefore runs about 31% fewer simulations.
- **How it changes play:**
  - Mid-turn, about half the evidence at the root comes from the previous search, which was deciding the previous
    move. Consecutive plays within a turn become more consistent: the agent follows through on its plan instead of
    re-deciding with fresh noise.
  - Turn-start decisions are unaffected.
  - Absolute win rate and HP may shift slightly.
- **Status.** Off in the card_marginals configs (user's call): the policy change is unmeasured. Run a paired
  quality A/B before turning it back on.
- **Bug (fixed).** With reuse, `legal_index` (`agents/combat/search/teacher_leaves.cpp`) mapped the chosen root edge through
  `particles.front()`, which after a rebase is a different particle. A draw-pile selection (Secret Technique /
  Weapon) then named the wrong card, or crashed with `public draw-selection action has no semantic match`. It now
  matches the edge's `semanticKey`; other actions are unchanged.

## Future paths (not implemented)

The estimates are % of CPU and come from the profile or recorded search data, not from measured
implementations.

| Idea | Estimate | Identical? | Notes |
|---|---|---|---|
| PGO (profile-guided optimization) on top of LTO | 10–20% (typical, unmeasured) | Yes | Needs a training run and a two-stage build |
| Check early stop every ~50 simulations instead of every 500 (`chunk`) | 2–5% of simulations (more with reuse) | Same moves; recorded `root_value` / `simulations_used` change | With the easy budget of 500 = one chunk, early stop never fires today |
| `stop_factor = 0.5` (existing setting) | ~10% of simulations | Near-identical moves (268/268 in the value-net test, `search_perf.md`) | Config only; lowers recorded `root_value` slightly, which doesn't matter where no rows are kept |
| Gauntlet worker: use the no-record `play_fight` | ~5% on cheap fights | Yes | One-line change; it builds rows and throws them away |
| Engine: the action queue uses `std::function` (heap allocations and copies) | ~5–8% | Yes | An engine refactor, not a quick win |
| Faster `SimpleAgent` rollout move / state hashing | up to ~10% / ~7% | Yes, if careful | Now the largest remaining non-engine parts |
| Build with `-march=native` | unknown | **No** (floating-point contraction changes the search) | Only with `-ffp-contract=off` |

## Reproducing

- The profiler, request builder, analysis scripts and raw results are in `scratch/profile_teacher/`:
  - `sampler.c`: `LD_PRELOAD` SIGPROF sampler.
  - `analyze.py`: self and inclusive time, and the `--children` / `--callers` views.
  - `make_requests.py`: card_marginals worker requests.
  - `budget_use.py`: where simulations go, from recorded `combat_v3` decision rows.
- Profile builds: `build/prof` (Release + `-g`), `build/lto`.
- To profile a worker:
  `LD_PRELOAD=$PWD/scratch/profile_teacher/sampler.so PROF_OUT=x.raw build/prof/card_marginals_worker REQ.json OUTDIR`,
  then `python3 scratch/profile_teacher/analyze.py x.raw`.
