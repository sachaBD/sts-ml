# slime-v8 deck split

Fixed once; every slime-v8 training and evaluation config uses it. One Slime Boss fight per bootstrap run
(`run_seed`), and every `run_seed` has a distinct deck, so the split is by `run_seed`. Resampled fights keep their
source `run_seed`, so they always fall on their source deck's side.

| part | rule (Slime Boss, `source_episode_id is null`) | decks |
|---|---|---:|
| **dev** | `id = 'act1-a20-8' and run_seed % 10 in (0, 1)` | 999 |
| **confirm** | `id = 'act1-a20-8' and run_seed % 10 in (2, 3)` | 972 |
| **train** | `id in ('act1-a20', 'act1-a20-1', 'act1-a20-2', 'act1-a20-5', 'act1-a20-6', 'act1-a20-7')` or `(id = 'act1-a20-8' and run_seed % 10 >= 4)` | 781 + 2,958 = 3,739 |

- dev: model/search selection. confirm: untouched until a candidate is chosen. Neither is ever trained on or
  resampled for training.
- The v6 156-fight benchmark (gen0's validation fights) falls in **train**: slime-v8 checkpoints may train on it,
  so it is no longer a benchmark for them. gen0 has never seen dev or confirm.
- Teacher win rate / terminal value by part: dev 71.9% / 0.371, confirm 70.5% / 0.369, train(a20-8) 70.3% / 0.367,
  train(old) 69.9% / 0.363.
- Train-deck resamples: `configs/slime-v8-train-resample-x4.toml` (run id `combat_v3/<date>/slime-v8-train-resample-x4`);
  training queries add `id = 'slime-v8-train-resample-x4'` rows. Their decks are train decks by construction.
