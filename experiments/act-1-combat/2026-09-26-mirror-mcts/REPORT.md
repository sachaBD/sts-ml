# Results

## gen0: imitating the teacher (2026-09-26)

**Question:** can the current small network (63k parameters) learn every act-1 fight, or does it need to be bigger or
different?

**Answer: the current architecture is a viable shared baseline.** One net trained on all fights plays almost as
well as the teacher across encounters. On Slime Boss it matches the slime-only net's win count, with no detectable
value difference. There is no demonstrated architecture bottleneck yet; this does not rule out gains from more capacity.

| | gen0 vs teacher |
|---|---|
| All 4,810 dev fights | **−1.2 HP-eq per fight** (95% CI −1.5 to −0.8); wins 93.0% → 91.6% |
| Encounter-weighted estimate per act-1 run | about **−6.5 HP-eq**, mostly from the boss (−2.4) and the elites (−1.3) |
| Slime Boss | −2.9 HP-eq; wins 74.7% → 69.9% |
| Elites | lagavulin −2.2, sentries −1.9, gremlin nob +1.0 |
| Hallway fights | −0.2 to −1.3 HP-eq each; almost no extra deaths |
| Slime Boss, vs the slime-only net | −0.1 HP-eq (CI −1.6 to +1.4); both won 698 of 999 |

**Where the gap is:** in lost fights, not in HP. On fights both players win, HP lost is within about 0.7 HP
everywhere. The net loses a few more boss and elite fights than the teacher. That matches the slime-only net, which
self-play then fixed.

**Training:** 1.82M positions, 12 epochs, 45 min on one CPU thread. Nearly all of the learning happened in the first
epoch, and there was no overfitting. The run is `value_net_v1/2026-09-26/act1-gen0`.

**Runs:** `combat_v3/2026-09-26/act1-teacher-dev` (teacher, 35 min) and `combat_v3/2026-09-26/act1-gen0-dev`
(gen0, 76 min, before the search speed-up). The confirm set is untouched.

Per-encounter table: `results/gen0-vs-teacher.md`.

## gen1: one round of self-play (2026-09-26)

**Question:** does one round of self-play fix the boss and elite losses that gen0 had, as it did on Slime Boss alone?

**Answer: clearly on Slime Boss; elite results remain uncertain.** gen1 beats the teacher on the aggregate benchmark,
and gen0 by a wide margin. The advantage over the teacher comes from Slime Boss; elsewhere it broadly matches the
teacher. Lagavulin's previous deficit is no longer evident, while Three Sentries retains a negative point estimate.

| | gen0 vs teacher | **gen1 vs teacher** | gen1 vs gen0 |
|---|---|---|---|
| All 4,810 dev fights, HP-eq per fight | −1.2 (−1.5, −0.8) | **+0.6 (+0.3, +0.9)** | +1.8 (+1.4, +2.1) |
| Wins | 93.0% → 91.6% | **93.0% → 93.5%** | 91.6% → 93.5% |
| Encounter-weighted estimate per act-1 run, HP-eq | −6.5 (−8.3, −4.7) | **+2.4 (+0.7, +4.1)** | +8.9 (+7.2, +10.7) |

Values in brackets are 95% CIs. Positive means the first player named is better.

### Where the difference comes from (HP-eq per act-1 run)

Each encounter's result is weighted by its frequency in the stored training runs. This is an isolated-fight utility
estimate, not measured HP gain or survival improvement in played-through acts; changed outcomes alter later states.

| part of the run | gen0 vs teacher | gen1 vs teacher | gen1 vs gen0 |
|---|---:|---:|---:|
| Boss (Slime Boss) | −2.4 | **+2.7** | +5.1 |
| Elites | −1.3 | −0.4 | +0.9 |
| Hallway fights | −2.9 | +0.1 | +3.0 |
| Events | 0.0 | 0.0 | 0.0 |
| **Total** | **−6.5** | **+2.4** | **+8.9** |

> 📊 **Chart placeholder:** stacked or grouped bars of HP-eq per run by part (boss / elites / hallway), one group each
> for gen0 vs teacher and gen1 vs teacher. Shows the boss contribution going from negative to positive.

### Boss and elites

| encounter | fights | win %: teacher / gen0 / gen1 | gen1 vs teacher, HP-eq per fight | gen1 vs gen0, HP-eq per fight |
|---|---:|---|---|---|
| Slime Boss | 999 | 74.7 / 69.9 / **78.3** | **+3.3** (+1.9, +4.7) | **+6.2** (+4.8, +7.6) |
| Gremlin Nob | 250 | 94.8 / 96.4 / 95.2 | +0.6 (−0.6, +1.7) | −0.4 (−1.4, +0.5) |
| Lagavulin | 250 | 92.4 / 88.4 / 91.6 | −0.1 (−1.8, +1.6) | **+2.1** (+0.5, +3.7) |
| Three Sentries | 250 | 90.4 / 87.2 / 87.2 | −1.5 (−3.2, +0.2) | +0.4 (−1.1, +1.9) |

- **Slime Boss:** gen1 wins 93 fights the teacher loses, and loses 57 that the teacher wins (McNemar p = 0.004).
  On fights both win, it also keeps more HP: it loses 18.3 HP per fight against the teacher's 20.4.
- **Lagavulin:** back to the teacher's level. gen0's extra losses (1 fight won only by gen0, 11 won only by the
  teacher) are gone (6 / 8).
- **Three Sentries:** a possible remaining gap. It has the same win rate as gen0 and loses 14 fights the teacher wins,
  against 6 the other way (p = 0.12). Worth monitoring, but not an established deficit or architecture failure.
- **Gremlin Nob:** no real difference among the three players.

> 📊 **Chart placeholder:** grouped bars of win % for teacher / gen0 / gen1 on Slime Boss and the three elites.

> 📊 **Chart placeholder:** forest plot of gen1 vs teacher HP-eq per fight with 95% CIs, one row per encounter (boss
> and elites at the top). Shows at a glance that only Slime Boss is clearly different.

### Hallway fights and events

- **Hallway fights:** gen1 matches the teacher on every encounter, within −0.6 to +0.8 HP-eq per fight, with at most
  2 extra deaths per encounter. It is clearly better than gen0 on most of them, by about +0.5 to +2 HP-eq, because it
  loses less HP.
- **Events:** each one has only 25–89 fights, so the results are noisy. event/lagavulin_event is −2.9 against gen0
  (31 fights, one extra loss), which is not a reliable signal.

### How gen1 was made

1. **Self-play:** gen0 replayed 4,076 train fights (`act1-a20-8`, `run_seed % 10 = 4`) with new randomness. It won
   3,787 of them. Run: `combat_v3/2026-09-26/act1-selfplay-gen1` (19 min).
2. **Fine-tune:** training started from gen0's weights on 52.5k positions from those fights, labelled with how each
   fight actually ended. Validation error was lowest after epoch 1, and that is the checkpoint kept. Later epochs
   overfit slightly (validation MSE 0.0071 → 0.0075). Run: `value_net_v1/2026-09-26/act1-gen1` (1 min).
3. **Evaluation:** the same 4,810 dev fights and search settings as gen0. Run:
   `combat_v3/2026-09-26/act1-gen1-dev` (31 min).

The confirm set is untouched.

**Caveats:** this is one round of self-play, evaluated on dev only. The differences on elites are within noise at 250
fights per encounter. The best checkpoint came from epoch 1, which suggests the self-play set is small for 10 epochs.
More self-play fights, or fewer epochs, are worth trying in gen2. CIs bootstrap fights rather than source runs
(and combine encounters independently), so they do not account for within-run dependence and may be too narrow.

Per-encounter tables: `results/gen1-vs-teacher.md`, `results/gen1-vs-gen0.md`.
