# slime-v8 report

Definitions:
- **Rollout teacher:** MCTS with guided-rollout leaves, 20k simulations, 8 belief particles. The target to beat.
- **Big-data network:** value network trained with slime-v6-gen0's recipe (blend 0.5 labels), from scratch on
  3,739 train decks instead of 625.
- **Dev fights:** 999 Slime Boss decks never trained on (`SPLIT.md`). Random move off for every player.

## Step 1–4: big-data network vs rollout teacher

Both players use the same search budget (20k simulations, 8 particles); only the leaf evaluation differs
(value network vs guided rollout). Paired over the same 999 fights; CI is a paired percentile bootstrap (20,000
resamples, seed 0).

| player | wins / 999 | mean terminal value | seconds/fight |
|---|---:|---:|---:|
| rollout teacher | 746 | 0.392 | 13.4 |
| big-data network | 698 | 0.371 | 16.5 |

- Difference (network − teacher): **−0.021 (95% CI −0.033, −0.010)**.
- Fights won by only the network / only the teacher: 62 / 110 (McNemar p < 0.001).
- The big-data network is **clearly weaker** than the rollout teacher. The gap is about the same size as
  slime-v6's small-data network vs teacher (−0.027 on 156 fights, 15k simulations), so 6× more teacher data did
  not visibly close it. (That comparison is on different fights and budgets; it is context, not a test.)
- Training: validation MSE 0.0077 (constant predictor 0.047); it reached that level after one epoch and later
  epochs mostly fit the training set.

Runs: `value_net_v1/2026-09-25/slime-v8-big-data`, `combat_v3/2026-09-25/slime-v8-big-data-dev`,
`combat_v3/2026-09-25/slime-v8-rollout-teacher-dev`. Numbers: `big-data-vs-rollout-teacher.json`
(`compare.py`).

## Generation 1: self-play fine-tune — beats the rollout teacher

The big-data network played 982 train decks (act1-a20-8 `run_seed % 10` in (4, 5), one new version each: new
combat RNG, starting HP ~ N(stored, 10)) with its dev search (20k simulations, 8 particles, no random moves): 641
wins, mean terminal value 0.355. It was then fine-tuned on those fights' decision rows, each labelled with its
fight's terminal value (4 epochs, lr 1e-4, fresh 10% validation split), and played the 999 dev fights.

| player (999 dev fights) | wins | mean terminal value |
|---|---:|---:|
| rollout teacher | 746 | 0.392 |
| big-data network | 698 | 0.371 |
| **generation 1 network** | **800** | **0.421** |

- vs big-data network: **+0.050 (95% CI +0.040, +0.062)**; wins only gen 1 / only big-data 128 / 26.
- vs rollout teacher: **+0.029 (95% CI +0.018, +0.040)**; wins only gen 1 / only teacher 101 / 47 (McNemar p < 0.001).
- First network player to beat the rollout teacher, on 999 fights no network trained on. **Promoted: current player.**
- Dev only; the 972 confirm fights are untouched. Gen 1 searches slower (20.3 s/fight vs teacher 13.4).

Runs: `combat_v3/2026-09-25/slime-v8-selfplay-gen1`, `value_net_v1/2026-09-25/slime-v8-gen1`,
`combat_v3/2026-09-25/slime-v8-gen1-dev`. Numbers: `gen1-vs-big-data.json`, `gen1-vs-rollout-teacher.json`.

## Generation 2

- Self-play (act1-a20-8 `run_seed % 10` in (6, 7), player gen 1): 770 / 995 wins, mean terminal value 0.422.
- vs current (gen 1): difference -0.0041 (95% CI -0.0121, +0.0038); wins only gen 2 / only gen 1 36 / 45.
- vs rollout teacher: difference +0.0251 (95% CI +0.0143, +0.0359); wins only gen 2 / only teacher 101 / 56.
- Dev: 791 / 999 wins, mean terminal value 0.417. **Not promoted**; current player stays gen 1.

Runs: `combat_v3/2026-09-26/slime-v8-selfplay-gen2`, `value_net_v1/2026-09-26/slime-v8-gen2`,
`combat_v3/2026-09-26/slime-v8-gen2-dev`. Numbers: `gen2-vs-current.json`, `gen2-vs-rollout-teacher.json`.

## Generation 3

- Self-play (act1-a20-8 `run_seed % 10` in (8, 9), player gen 1): 771 / 981 wins, mean terminal value 0.423.
- vs current (gen 1): difference -0.0064 (95% CI -0.0147, +0.0020); wins only gen 3 / only gen 1 46 / 54.
- vs rollout teacher: difference +0.0228 (95% CI +0.0121, +0.0336); wins only gen 3 / only teacher 103 / 57.
- Dev: 792 / 999 wins, mean terminal value 0.415. **Not promoted**; current player stays gen 1.

Runs: `combat_v3/2026-09-26/slime-v8-selfplay-gen3`, `value_net_v1/2026-09-26/slime-v8-gen3`,
`combat_v3/2026-09-26/slime-v8-gen3-dev`. Numbers: `gen3-vs-current.json`, `gen3-vs-rollout-teacher.json`.

## Generation 4

- Self-play (act1-a20, -1, -2, -5, -6, -7, player gen 1): 604 / 781 wins, mean terminal value 0.416.
- vs current (gen 1): difference +0.0000 (+3.7e-5; 95% CI -0.0085, +0.0086); wins only gen 4 / only gen 1 44 / 49.
- vs rollout teacher: difference +0.0292 (95% CI +0.0189, +0.0395); wins only gen 4 / only teacher 98 / 49.
- Dev: 795 / 999 wins, mean terminal value 0.422. **Promoted** (difference > 0): current player = gen 4.

Runs: `combat_v3/2026-09-26/slime-v8-selfplay-gen4`, `value_net_v1/2026-09-26/slime-v8-gen4`,
`combat_v3/2026-09-26/slime-v8-gen4-dev`. Numbers: `gen4-vs-current.json`, `gen4-vs-rollout-teacher.json`.
