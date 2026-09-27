# Act-1 combat with all three bosses: does the network beat the MCTS teacher?

Night of 2026-09-26/27. Ironclad, Ascension 20, act 1 fights only.

## In one paragraph

We trained one value network for every act-1 fight, including all three bosses (Slime Boss, The Guardian,
Hexaghost). The network plays with the same MCTS search as the teacher, but uses the network at the leaves instead of random
rollouts. **gen0** (trained to imitate the teacher) already plays as well as the teacher. **gen1** (gen0 after
one round of self-play) is **better than the teacher: +1.6 HP-eq per fight overall**. The gain is clearest on Slime
Boss (+5.5) and Gremlin Nob (+2.3). On The Guardian it is slightly positive, and on Hexaghost it is level. Both
are within noise. gen2 was planned but not run (out of time).

## Plan and status

| # | step | what | complete | actual time |
|---|---|---|---|---:|
| 1 | `ab-gen0` train | v2 small network, GPU, label `shift`, aux heads; buckets 6–9, ≤ 1,300 fights per encounter | ✅ yes | 5 min |
| 2 | `ab-teacher-dev` | MCTS teacher (rollout leaves, 20k sims, 8 particles, no random move) plays the 1,000 dev fights | ✅ yes | 15 min |
| 3 | `ab-gen0-dev` | gen0 plays the same 1,000 fights, same search | ✅ yes | 16 min |
| 4 | `ab-selfplay-gen1` | gen0 self-play on bucket 4 (unseen runs): every boss fight ×2, others ≤ 330 per encounter | ✅ yes | 69 min |
| 5 | `ab-gen1` fine-tune | from gen0, decision rows, real fight outcome as label | ✅ yes | 1 min |
| 6 | `ab-gen1-dev` → **gen1 report** | gen1 plays the 1,000 dev fights; this report | ✅ yes | 18 min |
| 7 | `ab-selfplay-gen2` | the better of gen0/gen1 self-plays bucket 5 | ❌ no | — |
| 8 | `ab-gen2` fine-tune | on both self-play sets | ❌ no | — |
| 9 | `ab-gen2-dev` → gen2 report | | ❌ no | — |
| 10 | endless self-play | best network keeps generating self-play (`forever.sh`, written, never run) | ❌ no | — |

Steps 1–6 took about 2.5 h. The run stopped after step 6 because the orchestrating agent ran out of tokens.
The machine then sat idle; it was not a compute failure. The configs for steps 7–9 are in `configs/`.

## Result

All players played the **same 1,000 held-out fights** from identical starting states, with the same search
(20k simulations, 8 belief particles, no random moves).

| player | wins / 1,000 | mean fight score | seconds per fight |
|---|---:|---:|---:|
| MCTS teacher (random-rollout leaves) | 771 | 0.385 | 7.8 |
| gen0 (imitates the teacher) | 777 | 0.387 | 8.2 |
| **gen1** (gen0 + one round of self-play) | **781** | **0.397** | 9.1 |

**How to read HP-eq:** each fight's score is converted to HP. One HP-eq is one HP kept, a potion kept is worth 4,
and a death costs your remaining HP plus 35. Positive means the first player is better. Brackets are 95% confidence
intervals. If one does not include 0, the difference is probably real.

### gen1 vs the teacher, by fight

| fight | fights | HP-eq per fight | wins: teacher → gen1 |
|---|---:|---|---|
| **Slime Boss** | 200 | **+5.5** (+1.9, +9.1) | 72.0% → 78.5% |
| The Guardian | 200 | +1.4 (−1.9, +4.5) | 70.5% → 71.5% |
| Hexaghost | 200 | −0.2 (−3.2, +2.7) | 54.5% → 53.5% |
| **Gremlin Nob** (elite) | 100 | **+2.3** (+1.1, +3.8) | 98% → 99% |
| Lagavulin (elite) | 100 | +0.2 (−2.3, +2.6) | 92% → 91% |
| Three Sentries (elite) | 100 | −0.7 (−3.6, +2.1) | 87% → 85% |
| hallway fights + events | 100 | pooled, see below | ~100% both |
| **all 1,000 fights** | 1,000 | **+1.6** (+0.3, +2.8) | 77.1% → 78.1% |

### Summary of all three comparisons

"Per act-1 run" weights each fight type by how often a run meets it (a run has about 7 fights and one boss).

| | gen0 vs teacher | **gen1 vs teacher** | gen1 vs gen0 |
|---|---|---|---|
| per fight (all 1,000) | +0.3 (−0.8, +1.4) | **+1.6 (+0.3, +2.8)** | +1.3 (+0.2, +2.3) |
| per act-1 run: boss | +0.5 (−0.9, +2.0) | **+1.9 (+0.3, +3.4)** | +1.3 (−0.1, +2.7) |
| per act-1 run: elites | −0.3 (−2.0, +1.3) | +0.8 (−0.9, +2.4) | +1.1 (−0.4, +2.5) |
| per act-1 run: hallway + events | −0.4 (−5.0, +3.3) | +1.8 (−3.1, +5.9) | +2.2 (+0.5, +3.9) |
| **per act-1 run: total** | −0.2 (−5.3, +4.1) | **+4.4 (−1.0, +9.1)** | **+4.6 (+1.9, +7.2)** |

## What this means

1. **One small network can learn all three bosses.** gen0 matched the teacher on every boss. That was the open
   question going in.
2. **Self-play helps.** gen1 beats both gen0 and the teacher over all fights. Most of the boss gain is on Slime Boss.
   The Guardian and Hexaghost did not clearly improve. Hexaghost is the hardest boss: both players win only ~54%.
3. **Nothing got worse.** No fight type is clearly below the teacher. Three Sentries is the weakest spot (−0.7,
   within noise). The previous act-1 experiment (Slime Boss only) had the same weak spot.
4. **Caveat:** the per-run totals have wide intervals, because hallway fights are only 100 of the 1,000 dev
   fights. The boss and per-fight numbers are the reliable ones. This is one evaluation on "dev" fights. The
   separate "confirm" fights were never touched and are kept for a final check.

## How it was done

**Data.** Teacher games from `combat_v3/2026-09-26/act1-all-bosses-a20-scaled-search`: 12,106 act-1 runs, 80,769
fights, all three bosses. Split by run so no deck is on two sides:

| part | runs (`run_seed % 10`) | used for |
|---|---|---|
| dev | 0, 1 | the 1,000 evaluation fights: 200 per boss, 100 per elite, 100 others |
| confirm | 2, 3 | untouched |
| self-play | 4 (gen1), 5 (planned gen2) | runs no network trained on |
| gen0 training | 6–9 | |

**gen0.** Network "v2, small" (width 64, one 128-wide residual head block, ~150k parameters), trained on a GPU
in 5 min. Training data was balanced: at most 1,300 fights per encounter, about the number of fights per boss, so
easy fights don't dominate. That gave 21k fights and 1.34M positions. Labels are a 50/50 mix of the teacher's
search value and the real fight outcome (`label = "shift"`). There are also two small side-outputs (win and HP) to
help learning.

- A quick sweep picked the small v2. The default v2 (600k parameters) overfitted after one epoch, and every size
  reached the same validation error. So we took the one that is cheapest inside the search.

**gen1.** gen0 played ~1,210 unseen runs (bucket 4): every boss fight twice (1,942 fights, 65% won), plus up to
330 fights of every other encounter (4,348 fights). gen0 was then fine-tuned for 1 minute on the positions it
played, each labelled with how that fight actually ended.

- The fine-tune's best checkpoint came at epoch 2 of 10, which suggests more self-play data would help.

## Next steps

1. **gen2** (configs ready): gen1 self-plays bucket 5, then is fine-tuned on both self-play sets. Use fewer epochs
   or more self-play fights, since gen1's fine-tune overfitted after epoch 2.
2. Keep generating self-play with the best network (`forever.sh`, not yet run).
3. Before any claim beyond this report, play the untouched confirm fights once with the final network and the
   teacher.

## Files and runs

- `results/gen1-vs-teacher.md`, `results/gen1-vs-gen0.md`, `results/gen0-vs-teacher.md`: full per-fight tables
  (+ `.json`). Made with `analyze.py`.
- `NOTEBOOK.md`: timeline, decisions and incidents. `configs/`: one file per run. `drive.sh` ran them.
- Networks: `value_net_v1/2026-09-26/ab-gen0`, `value_net_v1/2026-09-27/ab-gen1`.
- Games: `combat_v3/2026-09-26/ab-teacher-dev`, `…/ab-gen0-dev`, `combat_v3/2026-09-27/ab-gen1-dev`,
  `combat_v3/2026-09-27/ab-selfplay-gen1-boss`, `…/ab-selfplay-gen1-rest`.
- Previous experiment (Slime Boss the only boss): `../2026-09-26-mirror-mcts/`.
