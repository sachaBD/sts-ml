# App spec: `apps/compare_fights/`: paired comparison of two `combat_v3` runs

## Purpose

Compare how two agents did on the **same fights**, using only stored data. Each side is a
`combat_v3` run. The app doesn't care which agent played either side: teacher vs value-net teacher
(`value_play`), gen1 vs gen0, two budgets, ...

## Usage

```bash
./apps/compare_fights/run.sh apps/compare_fights/slime-value-4.toml [--scratch]
```

```toml
[run]
id = "slime-value-4-vs-teacher"
baseline = ["combat_v3/2026-09-23/act1-a20", "combat_v3/2026-09-23/act1-a20-1"]
candidate = "combat_v3/2026-09-23/slime-value-4-play"
```

Fights = the candidate's `episode_id`s (one fight each). `baseline` and `candidate` each accept
one or more `combat_v3` run ids. Source rows are merged by `episode_id`; duplicate episode ids across
sources fail to avoid ambiguous provenance. Baseline fights outside the candidate set are ignored, and
a candidate fight missing from the baseline fails the run.

## Data

One row per fight from each run: the `decision` rows, grouped by `episode_id`:
`won`, `final_hp`, `potions`, `terminal_value`, `starting_hp`, `starting_max_hp`, number of
decisions, max `turn`, total `simulations_used`. Joined on `episode_id`.

Sanity check: `starting_hp` and `starting_max_hp` must match per fight (same fight). Mismatches fail
the run.

## Statistics (paired, per fight)

| metric | reported | test |
|---|---|---|
| win rate | each side; fights won only by baseline / only by candidate | exact McNemar (two-sided binomial on the discordant fights) |
| `terminal_value` (primary) | mean each side; mean paired difference, 95% CI | Wilcoxon signed-rank; paired bootstrap CI (10,000 resamples, fixed seed) |
| `final_hp`, `potions`, turns | mean each side; mean paired difference, 95% CI | same as `terminal_value` |
| compute | simulations per decision, decisions per fight | descriptive |

- **Primary metric:** `terminal_value`. It is the training target's scale, and win rate saturates.
- **Also report:** better / worse / equal counts on `terminal_value`.
- **No multiple-comparison correction:** secondary metrics are descriptive and labelled as such.
- **n is stated up front:** small candidate runs (e.g. 14 validation fights) give wide CIs.

## Report

`out/report.md`, also printed to stdout. Short, and in this order:

1. **Header:** baseline and candidate run ids, their inputs, the number of fights compared.
2. **One-line verdict on the primary metric:** mean difference, CI, p.
3. **Table:** the metrics above.
4. **Worst regressions:** the 5 fights where candidate is furthest below baseline on
   `terminal_value` (episode_id, encounter, both values, both final HP), to inspect by hand.

## Output: new schema `fight_comparison_v1`

```
out/report.md
out/summary.json     config, n, per-metric {baseline_mean, candidate_mean, diff, ci_low, ci_high, p}
out/pairs.parquet    one row per fight: episode_id, encounter, then each metric as baseline_* / candidate_*
```

`inputs` in `run.json` = `[baseline, candidate]`. Add a schema row to `runs/README.md`.

## Layout

```
apps/compare_fights/
  run.sh, job.sh
  compare.py          load, pair, test, write report
  slime-value-4.toml
```

## Dependencies

scipy (Wilcoxon, binomial test), added to `requirements.txt`.

## Not in scope (v1)

- Per-decision agreement (does the candidate pick the teacher's move at the teacher's states).
  That needs the candidate searched at the stored states, which is a different app.
- More than two runs at once.
