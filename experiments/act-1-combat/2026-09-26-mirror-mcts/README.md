# Act-1 combat: one value net for every fight

**Goal:** a single neural combat player for all act-1 fights (A20 Ironclad). Card picks and pathing are not part of
this; they come from replaying the stored runs.

**The player:** the MCTS teacher search, with the random rollouts at its leaves replaced by a value network.

**The recipe** is the one that worked on Slime Boss (`experiments/slime-v8/REPORT.md`):
1. Train the net to imitate the teacher (gen0). It ends up slightly weaker than the teacher.
2. Let it play itself and fine-tune on the real fight outcomes (gen1, gen2, ...). On slime this beat the teacher.

## Status

| step | status | result |
|---|---|---|
| gen0: imitate the teacher on all act-1 fights | done | Works. The net trails the teacher by about 1 HP per fight, and no encounter is worse than the slime specialist. See `REPORT.md` |
| gen1: self-play + fine-tune | done | Beats the teacher by +2.4 HP-eq per act-1 run and gen0 by +8.9. Slime Boss is clearly better than the teacher; Three Sentries still trails it slightly. See `REPORT.md` |

## Files

```
README.md              this file
REPORT.md              results so far (REPORT.pdf: the same, as a PDF)
configs/               one file per run (commands inside each file)
  act1-gen0.toml         train gen0
  act1-teacher-dev.toml  teacher plays the dev fights (the baseline, run once)
  act1-gen0-dev.toml     gen0 plays the dev fights
  act1-selfplay-gen1.toml  gen0 plays train fights (self-play data for gen1)
  act1-gen1.toml           fine-tune gen0 into gen1
  act1-gen1-dev.toml       gen1 plays the dev fights
split.py               defines train / dev / confirm; writes dev_fights.csv and results/split_tables.md
analyze.py             compares two players on the same fights, per encounter
dev_fights.csv         the 4,810 fights every player is evaluated on
results/               analyze.py output tables
```

## How it works

**Data.** The teacher's own games: 6,895 act-1 runs, 46k fights (`combat_v3` runs `act1-a20*`).

**Split.** Whole runs are split, so no deck appears on both sides:

| part | runs | used for |
|---|---:|---|
| train | 4,518 | training and self-play |
| dev | 1,198 | evaluation |
| confirm | 1,179 | untouched, kept for a final check |

**Evaluation.** Every player plays the same 4,810 dev fights: all 999 Slime Boss fights, plus up to 250 of every
other encounter. Starting states are identical, and every player uses the same search budget (20k simulations, no
random moves). Results are paired fight by fight against the teacher.

**Unit: HP-eq.** This is the difference in fight score, converted to HP. One HP-eq is one HP; a potion is worth 4, and
a death costs all remaining HP plus 35. Negative means worse than the teacher.

## Reproduce

Run from the repo root, one config at a time. The command is on the first line of each config.

```bash
./apps/value_train/run.sh experiments/act-1-combat/2026-09-26-mirror-mcts/configs/act1-gen0.toml
./apps/value_play/run.sh  experiments/act-1-combat/2026-09-26-mirror-mcts/configs/act1-teacher-dev.toml
./apps/value_play/run.sh  experiments/act-1-combat/2026-09-26-mirror-mcts/configs/act1-gen0-dev.toml
PYTHONPATH=python .venv/bin/python experiments/act-1-combat/2026-09-26-mirror-mcts/analyze.py \
    --candidate combat_v3/2026-09-26/act1-gen0-dev \
    --teacher combat_v3/2026-09-26/act1-teacher-dev combat_v3/2026-09-25/slime-v8-rollout-teacher-dev
```

gen1 (about 1 hour in total), in order:

```bash
./apps/fight_resample/run.sh experiments/act-1-combat/2026-09-26-mirror-mcts/configs/act1-selfplay-gen1.toml
./apps/value_train/run.sh    experiments/act-1-combat/2026-09-26-mirror-mcts/configs/act1-gen1.toml
./apps/value_play/run.sh     experiments/act-1-combat/2026-09-26-mirror-mcts/configs/act1-gen1-dev.toml
PYTHONPATH=python .venv/bin/python experiments/act-1-combat/2026-09-26-mirror-mcts/analyze.py \
    --candidate combat_v3/2026-09-26/act1-gen1-dev \
    --teacher combat_v3/2026-09-26/act1-teacher-dev combat_v3/2026-09-25/slime-v8-rollout-teacher-dev
PYTHONPATH=python .venv/bin/python experiments/act-1-combat/2026-09-26-mirror-mcts/analyze.py \
    --candidate combat_v3/2026-09-26/act1-gen1-dev --teacher combat_v3/2026-09-26/act1-gen0-dev
```

## Next: gen2 (not configured)

gen1 beat gen0, with the gain mainly on Slime Boss and Lagavulin (see `REPORT.md`). What's left is Three Sentries,
which still trails the teacher slightly. The fine-tune's best checkpoint came from epoch 1, so gen2 should use more
self-play fights or fewer epochs.
