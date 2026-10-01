# How to compare combat agents

The rules for claiming "agent A plays fights better than agent B". They come from the elite-bench study
(`experiments/elite-bench/`): a single-run comparison on a few hundred fights is mostly noise.

## 1. Score: HP-eq, plus wins

- **HP-eq per fight** = `terminal_value × (55 + max HP)`: on a win, 35 + final HP + 4 per potion kept; on a death, 0.
  A death costs your HP plus 35, and one potion is worth 4 HP.
- **Report** the mean paired difference in HP-eq, win rates (or win probabilities, §3) and discordant wins
  (fights only one agent won) **per elite and overall**.
- **Why wins as well:** HP-eq can rise through HP saved in easy fights while deaths in hard fights get worse.
  v3-t2 was +0.75 HP-eq overall but −4 on the Sentries fights that decide survival.

## 2. Pair everything

- Both agents play **the same stored fight starts**: same deck, HP and game RNG (`value_play` replays them).
  Analyse per-fight differences, never two separate averages. Pairing removes the deck-to-deck variance, which is
  about 90% of the total.
- Use the same simulator for both. Replays that no longer match the stored fight are skipped
  (`skip_diverged = true`); both agents skip the same ones, so pairing still holds.

## 3. Several search seeds per fight (`search_salt`)

- The search is deterministic given the state, so one run gives **one outcome per fight**. Tiny changes (another
  net, another seed) send the fight down a different path. Whether the agent lives or dies at the margin is then
  close to a coin flip.
- Run each agent with **K salts** (`search_salt = 0 … K−1`; 0 reproduces the historical runs bit for bit). Score each
  fight by its **mean over salts**, i.e. an expected HP-eq and a win probability.
- K = 4 cut the SD of the per-fight difference from 11.8 to 7.9 HP-eq and the fights needed by ~2.2×. K = 2 gets
  most of that.

## 4. Know the noise floor (A/A test)

Play **the same agent twice with different salts** and compare the runs as if they were two agents. Measured
at 20k simulations on 1,676 elite fights:

| | MCTS vs MCTS | v3 vs v3 | MCTS vs v3 |
|---|---:|---:|---:|
| SD of per-fight Δ HP-eq | 10.5 | 9.5 | 11.8 |
| fights where the outcome flips | ~4% | ~3% | ~5% |

A difference in deaths no larger than the A/A column is chance. Re-run the A/A test whenever the search changes.

## 5. Intervals

- 95% **cluster bootstrap over source `run_seed`** (10,000 resamples, percentile). Fights from one run share a
  deck, so they aren't independent.
- A result counts only if its interval excludes 0. Point estimates alone don't.

## 6. Sample size

Paired fights needed to detect a true difference (80% power, two-sided 5%; SDs from §3):

| salts per fight | 0.5 HP-eq | 1.0 HP-eq | 2.0 HP-eq |
|---|---:|---:|---:|
| K = 1 | ~4,400 | ~1,100 | ~275 |
| K = 4 | ~2,000 | ~500 | ~125 |

The old 150-fights-per-elite validation could only detect about 2.7 HP-eq per elite, while real effects are
about 1. Size runs from this table before launching them.

## 7. Fight sets and selection bias

- **Split by run seed** (`run_seed % 10`) so training and evaluation never share a deck. Choose fights by hash
  (never by outcome) and freeze the list in a CSV before playing.
- **Picking the best of N arms on a set inflates its score.** v3-t2: +0.82 on validation, +0.42 on the final set.
  t5: +1.37 on the benchmark, +0.83 on fresh fights. So:
  1. choose models and settings on a dev set;
  2. **write the success rule down before** the confirmation run;
  3. confirm once on fresh fights nobody chose anything on.
- Current sets (buckets 0–1 of `act1-all-bosses-a20-scaled-search`): `experiments/elite-bench/bench_fights.csv` (dev
  benchmark, 600 per elite; now used for selection) and `confirm_fights.csv` (used once, for t5). The next claim
  needs a fresh cohort.

## 8. Strata that matter

- **Unwinnable fights** (the oracle, which sees draws and RNG, loses): about 3%. Report results with and without
  them.
- **Pivotal fights** (some fair run lost, but the fight is winnable): about 12%. Nearly all differences in deaths
  happen here.
  - They are chosen *using* outcomes, so evaluate them only on **held-out salts** (e.g. choose on salts 0–3,
    score on 4–7).
  - Selection flatters the runs it picked from: MCTS's win rate on the pivotal set moved 53.7% → 57.9% from the
    selection salts to fresh ones, which is regression to the mean.
- Always show each elite separately. An overall gain with a clear loss on one elite isn't a win.

## 9. Checklist for a result

1. Same fights and starts; same simulator; skipped-fight count stated.
2. K ≥ 2 salts per agent (K ≥ 4 for claims); A/A floor quoted if deaths are discussed.
3. Paired HP-eq with cluster-bootstrap 95% CI, per elite and overall; win probabilities and discordant wins.
4. Pivotal stratum on held-out salts.
5. Say whether the fights were used for selection. Claims come only from a pre-registered run on fresh fights.

## Tools and references

- Configs: `experiments/elite-bench/make_configs.py`. Runner: `run.sh`. Analysis (all sections above):
  `PYTHONPATH=python .venv/bin/python experiments/elite-bench/analyze.py`. Pivotal list: `pivotal.py`. Two-run
  paired table: `experiments/elite-v3/compare.py`.
- Results: `experiments/elite-bench/results/analysis-A.md` (headroom, A/A, v3 vs MCTS, power),
  `analysis-R2.md` (round-2 nets, pivotal held-out), `confirm.md` (fresh-fight confirmation), `analysis.md`
  (final, incl. budget and particle curves); plan and log in `PLAN.md` / `LOG.md`.
- Background: `experiments/act-1-combat/2026-09-28-elite-specialist/`, `experiments/elite-v3/REPORT.md`,
  `experiments/act-1-easy-combats/README.md` (oracle and budget methods on easy fights).
