# act-1 combat: one value net for every act-1 fight

A20 Ironclad, act 1 fights only; card picks and routing come from the existing run replay. The player is the MCTS
teacher search with a value net at the leaves (`apps/value_play`, `apps/fight_resample`).

## Status

| step | state | result / run |
|---|---|---|
| 1. gen0: imitate the teacher on all act-1 fights | **done** | Topology is sufficient: the gap to the teacher is within the slime specialist's on every encounter. `REPORT.md` |
| 2. gen1: self-play, fine-tune on terminal value | **proposed, not started** | see "Next: gen1" below |

## Files

| file | what |
|---|---|
| `README.md` | this: setup, split, protocol, status, plan |
| `REPORT.md` | gen0 results |
| `split.py` → `split_tables.md`, `dev_fights.csv` | builds the split and the frozen dev eval fights |
| `analyze.py` | paired per-encounter comparison of a player vs a baseline, in HP-eq (`gen0-vs-*.md/json`) |
| `configs/act1-gen0.toml` | gen0 training (`apps/value_train`) |
| `configs/act1-teacher-dev.toml` | teacher replay of the non-boss dev fights (the baseline; run once, reused) |
| `configs/act1-gen0-dev.toml` | gen0 plays the dev fights |

## Units

- **Terminal value** (tv): win = (35 + final_hp + 4·potions) / (55 + max_hp), loss = 0.
- **HP-eq** = Δtv × (55 + max_hp), about Δtv × 135. One HP-eq is one HP. A potion is 4 HP-eq, and a death costs
  all remaining HP plus 35.
- **Reference gap:** the slime-v8 "big-data" specialist vs the teacher on 999 slime dev fights is −2.8 HP-eq
  (−4.3, −1.3), with wins 698 vs 746 (`rundecks/slime-v8/REPORT.md`). Imitation nets are expected to trail the
  teacher by about this much; self-play is what lifts them above it.

## Data and split (fixed; `split.py`)

The data is the existing teacher bootstrap runs `combat_v3` `act1-a20`, `-1`, `-2`, `-5`, `-6`, `-7`, `-8`: 6,895
full act-1 runs, 46,156 fights. The teacher used 15k sims and one random move per fight. Resamples are excluded
(`source_episode_id is null`), and there are no oracle rows.

The split is by run (`run_seed`), so a deck never appears on two sides. It is the slime-v8 rule
(`rundecks/slime-v8/SPLIT.md`) extended to all fights, so the slime parts are identical and slime-v8 runs on dev
can be reused.

| part | rule | runs | fights | use |
|---|---|---:|---:|---|
| **train** | everything not below | 4,518 | 30,206 | gen0 training. Self-play generations draw decks from `act1-a20-8`, `run_seed % 10 >= 4` |
| **dev** | `id = 'act1-a20-8' and run_seed % 10 in (0, 1)` | 1,198 | 8,104 | gameplay eval, model selection |
| **confirm** | `id = 'act1-a20-8' and run_seed % 10 in (2, 3)` | 1,179 | 7,846 | untouched until a final candidate |

- **Dev eval set** (`dev_fights.csv`): 4,810 fights. It has all 999 Slime Boss dev fights plus the first 250 dev
  fights by `episode_id` for every other encounter (all of them where there are fewer). 250 paired hallway fights
  give about ±1 HP.
- **Training mix:** natural, no reweighting. The boss is 24% of rows, sentries 11%, and the other encounters 0.3–8%.
  Every encounter fit fine (validation R² 0.81–0.95), so no reweighting is needed.
- **Events** (25–89 dev fights each) are reported descriptively only.

## Evaluation protocol (same as slime-v8)

- Both players use MCTS with 20k sims, 8 particles and **no random move**, on the same dev fights with the same
  start state and seed. The net uses value leaves; the teacher uses guided-rollout leaves.
- Baseline: `combat_v3/2026-09-26/act1-teacher-dev` (non-boss) plus
  `combat_v3/2026-09-25/slime-v8-rollout-teacher-dev` (boss).
- `analyze.py` reports, per encounter: the paired Δ HP-eq with a 95% bootstrap CI, win %, McNemar, HP lost on
  fights both won, and a per-run total weighted by encounter frequency.

Measured compute (10 workers): the teacher replay of 3,811 non-boss fights took 35 min, and the net on all 4,810
dev fights took 76 min. The net's search time splits boss 36%, hard 31%, elite 16%, easy 13%, event 3%.

## Next: gen1 (proposed; awaiting go)

This is slime-v8's expert-iteration step, applied to all act-1 fights:

1. **Self-play (`apps/fight_resample`, `value_run = act1-gen0`).** Each train fight is rebuilt at its start and
   played again by gen0 with a fresh combat RNG and starting HP ~ N(stored, 10). The search is the same as on dev:
   20k sims, 8 particles, no random move.
   - Decks: `act1-a20-8`, `run_seed % 10 = 4`, which is 4,076 fights (502 boss, 784 elite), about 1 h.
   - Every encounter is included. Hallway fights are about 2.7 of the 6.5 HP-eq per-run gap, and fine-tuning on a
     subset of encounters would risk regressing the rest.
2. **Fine-tune gen0 → gen1** on the self-play decision rows, labelled with terminal value.
   - lr 1e-4, weight decay 1e-4, batch 128, fresh 10% validation split.
   - 10 epochs, cosine decay, keep the best-validation checkpoint (about 60k rows, well under a minute per epoch).
3. **Eval** on the dev set (about 76 min). Compare with `analyze.py` against the teacher and against gen0.

In total that is about 2.3 h. If gen1 helps, gen2 adds the next deck bucket (`run_seed % 10 = 5`) and trains on the
accumulated self-play data, as slime-v8 gen2/gen3 did.
