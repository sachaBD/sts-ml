# Card-pick search: first build and first results (2026-09-30)

## What exists
- **`apps/card_search/worker.cpp`** (target `card_search_worker`, in `build/main`): the C++ search worker.
  - It plays Ironclad A20 Act 1 from a seed with the macro simulator (`apps/common/macro_sim`: common random numbers
    and content filter on), up to the j-th card-reward decision (the root).
  - From the root it runs N rollouts per option (each offered card, then skip) to the end of Act 1.
  - Rollout r shares its game seed and its fight random numbers across all options: common random numbers.
  - Fight results and card picks are requested from Python in batches (JSON lines over stdin/stdout).
- **`apps/card_search/search.py`**: the Python batch server.
  - **Fights:** sampled from the outcome-model ensemble (`tpair1-co2-w32-h64-l1-d30`, 3 seeds; one member per rollout)
    by inverse-CDF sampling with the worker's random numbers. HP is the bin center, clipped to max HP.
  - **Picks (policy v0):** the option with the highest outcome-model expected score (P(win) × E[HP]), averaged over the
    act boss (weight 2) and the three Act 1 elites, at current HP.
  - Everything else is SimpleAgent.
- **`experiments/card-search/agreement.py`**: the repeat-agreement analysis.

## Result 1: searches agree with themselves, and common random numbers work
Setup: 40 seeds × decisions {1, 3} × 2 independent repeats (same root, different rollout seeds) × 500 rollouts per
option; 4 options per decision. Data: `experiments/card-search/data/agree1-shard*.jsonl`. It took about 28 min on 8
processes.

| Rollouts per option | 25 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|
| Best option identical in both repeats (79 decisions; chance 25%) | 75% | 86% | 87% | 92% | **99%** |

- Correlation of the option values between repeats: r = 0.98.
- Spread between the best and worst option: median 4.2 HP-score (score = HP at the end of Act 1, 0 if dead).
- Common random numbers: standard error of (option − skip) is 0.32 paired vs 1.18 unpaired, a **13× variance
  reduction**.
- Search picks skip 8% of the time.

## Result 2: the simulator is optimistic
- **Sim vs real, same card policy:** SimpleAgent card picks in the simulator clear Act 1 **71.5%** of the time (200
  seeds, ±3.2) against **60.6% ± 1.5** in real games with the MCTS combat player. HP entering the boss is 57 (sim) vs a
  mean of about 53 (natural boss states).
- **Model-greedy picks look far better in the sim:** with v0 picks, simulated clear rates are 77–95%. Part of this is
  likely the policy **exploiting the outcome model** (v0 picks what the model likes, and the same model then judges the
  fights), so it is not evidence of better play.

## What this means
- **The search machinery works:** with 500 rollouts, the pick is stable, and common random numbers make that cheap.
- **Stability is not correctness.** Both repeats share the same simulator bias. The next evidence must be **real games**:
  search / v0 picks vs SimpleAgent picks, with the real combat player. The simulator's optimism must also be tracked
  (sim-vs-real gap) and reduced, e.g. by calibrating boss predictions, carrying HP more realistically, and retraining the
  outcome model on fights from the new policies.
