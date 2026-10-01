# Act-1 combat, all three bosses: gen0 → gen1 → gen2

**Results: `REPORT.md`** (gen1 beats the MCTS teacher; gen2 not run). Timeline and decisions: `NOTEBOOK.md`.
Note: the plan below was revised during the night (1,000-fight dev set, v2 small network, balanced data); the
configs in `configs/` and REPORT.md are authoritative.

**Goal:** one neural combat player for every act-1 fight (A20 Ironclad), now including **The Guardian and Hexaghost**.
The previous experiment (`../2026-09-26-mirror-mcts/`) only had Slime Boss as its boss. It showed that the recipe
(imitate the teacher → self-play fine-tune) beats the MCTS teacher there. This experiment repeats the recipe on the
new all-bosses dataset and answers:

1. Can one small net (63k params, width 64) imitate the teacher on all three bosses (gen0)?
2. Does self-play lift the net above the teacher on each boss, not only Slime Boss (gen1, gen2)?
3. Do elites and hallway fights stay at least level with the teacher?

**The teacher** (the player to beat): MCTS with guided-rollout leaves, 20k simulations, 8 particles, **no random
move**. The nets use the identical search with value-net leaves. `stop_factor` / `merge_identical_cards` are off for
every evaluated player (same protocol as mirror-mcts and slime-v8).

## Data and split

`combat_v3/2026-09-26/act1-all-bosses-a20-scaled-search`: 12,106 runs, 80.8k fights, 4.5M rows. The bootstrap teacher
used scaled budgets (500 / 2k / 5k / 5k / 15k sims for easy / hard / elite / event / boss), stop_factor 0.5,
merge_identical_cards, and one random move per fight. Split by run (`split.py`, `results/split_tables.md`):

| part | run_seed % 10 | runs | fights | rows | Slime / Guardian / Hexaghost fights |
|---|---|---:|---:|---:|---|
| train | 4–9 | 7,264 | 48.3k | 2.70M | 2,011 / 2,017 / 1,890 |
| dev | 0, 1 | 2,421 | 16.2k | 0.90M | 697 / 649 / 642 |
| confirm | 2, 3 | 2,421 | 16.3k | 0.91M | never touched |

**Dev eval subset** (`dev_fights.csv`): 6,614 fights = every dev boss fight (1,988) + the first 250 dev fights of
every other encounter. All players play exactly these fights from identical starting states. Results are paired fight
by fight (`analyze.py`, HP-eq, as in mirror-mcts). Per-run weights come from the train split.

## Decisions (owner may veto)

- **gen0 data: the full train split (2.70M rows).** Boss data is the limiting part (~2k train fights per boss, fewer
  than the 3.7k Slime Boss fights of the previous gen0), so nothing is subsampled. Memory is ~4 GB packed and the
  run takes about 1 h for 12 epochs. Recipe unchanged from act1-gen0 (blend 0.5), although the new data's
  hallway root values come from small, early-stopped searches.
- **gen1 self-play: train bucket 4** (~8.1k fights, twice the size of mirror-mcts's self-play set).
- **gen2:** gen1 plays more train buckets (5 and 6 if time allows), then is fine-tuned from gen1 on all self-play
  fights so far. The bucket count is fixed from measured gen1 timings to fit the 8 h budget.
- The teacher dev run uses the old `act1-gen0` as its value run for the train-fight guard only (weights unused;
  no overlap with the new dev fights, checked). This lets it run while ab-gen0 trains.

## Runs (in order; `drive.sh`)

| step | config | what |
|---|---|---|
| 1 | `ab-gen0` ∥ `ab-teacher-dev` | train gen0 while the teacher plays the dev fights |
| 2 | `ab-selfplay-gen1` | gen0 self-play, bucket 4 |
| 3 | `ab-gen1` | fine-tune gen0 → gen1 |
| 4 | `ab-gen1-dev`, `ab-gen0-dev` | gen1 and gen0 play the dev fights → gen1 report |
| 5 | `ab-selfplay-gen2`, `ab-gen2`, `ab-gen2-dev` | gen2 |

```bash
experiments/act-1-combat/2026-09-26-all-bosses/drive.sh par:ab-gen0+ab-teacher-dev ab-selfplay-gen1 ab-gen1 ab-gen1-dev ab-gen0-dev
PYTHONPATH=. .venv/bin/python experiments/act-1-combat/2026-09-26-all-bosses/analyze.py \
    --candidate combat_v3/<date>/ab-gen1-dev --teacher combat_v3/<date>/ab-teacher-dev --md results/gen1-vs-teacher.md
```
