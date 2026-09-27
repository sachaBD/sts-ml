# Notebook: all-bosses gen0 → gen1 → gen2 (running log, newest at the bottom)

Owner go-ahead 2026-09-26 ~23:00 UTC, ~8 h compute. Plan table: README.md (to be refreshed); configs in `configs/`;
`drive.sh STEP...` runs steps one at a time and logs to `results/drive.log`.

## Settled decisions
- Model: **v2** (`deep_sets_v2` defaults, ~603k params), GPU, batch 1024, gen0 label `shift` (0.5), aux won 0.01 /
  hp 0.1. Fallback v1 only if v2 fails. C++ v2 parity passed (commits 7caa65f / 7fc44a5).
- gen0 data: train buckets 6–9, ≤ 1,300 fights per (category, encounter), random by hash(episode_id).
  Buckets 4, 5 held out as unseen self-play runs (gen1, gen2). Dev = buckets 0, 1; confirm = 2, 3 (never touched).
- Dev eval: 1,000 fights (200 per boss, 100 per elite, 100 other), 20k sims, 8 particles, no random move, all players.
- Self-play per gen: every boss fight of the bucket ×2 samples, other encounters ≤ 330 fights ×1.
- One job at a time (RAM: value_play setup peaked ~10 GB; an earlier parallel launch nearly OOM'd and was killed).

## Log

- 23:0x start `ab-gen0` training.
- 22:57 `ab-gen0` (v2 default, lr 1e-3, wd 1e-4, 20 epochs, 34 s/epoch on GPU; 1.20M train / 137k valid rows;
  shift clipped 10.6k targets). **Overfits from epoch 1**: valid MSE 0.00539 → 0.00563 → 0.00571 → 0.00590 → 0.00591
  while train MSE 0.0043 → 0.0019. Killed at epoch 6 to try regularisation instead of burning 20 epochs.
  (Aux losses are small next to value loss after epoch 1 — aux_won 0.13×0.01, aux_hp 0.010×0.1 vs value 0.0083. OK.)
- 23:1x sweep, scratch runs, 5 epochs each: wd 0.05 / lr 3e-4 / small v2 (width 64, head 128×1, wd 0.01).
  **Incident:** the wd05 run was OOM-killed (exit -9): I ran a DuckDB check query while training loaded its data
  (training load peaks ~10 GB RSS). Rule from now: nothing runs next to a data load / value_play setup. wd05 re-queued.
- 23:3x sweep results (valid MSE by epoch; 5 epochs, cosine):
  | variant | e1 | e2 | e3 | e4 | e5 |
  |---|---|---|---|---|---|
  | default v2, lr 1e-3, wd 1e-4 (the killed ab-gen0) | 0.00539 | 0.00563 | 0.00571 | 0.00590 | 0.00591 |
  | default v2, lr 3e-4 | 0.00644 | 0.00539 | 0.00544 | 0.00540 | 0.00542 |
  | **small v2** (width 64, head 128×1), wd 0.01 | 0.00545 | 0.00534 | 0.00526 | **0.00518** | 0.00522 |
  wd 0.05 run cancelled (not needed). All sit near ~0.0053: the floor looks like label noise, not capacity.
  **Decision:** gen0 = small v2, wd 0.01, 6 epochs. Same accuracy and far cheaper inference inside a 20k-sim search.
  gen1/gen2 fine-tunes use the same [model] (required) and wd 0.01.
- Boss val MSE (small, e4): hexaghost 0.0065, slime 0.0108, guardian 0.0063 (baselines 0.027 / 0.042 / 0.032).
- 23:29 **ab-gen0 done** (5 min): `value_net_v1/2026-09-26/ab-gen0`, small v2, best epoch 4, valid MSE 0.00523
  (e1..e6: 0.00545 0.00548 0.00542 0.00523 0.00541 0.00538). Chain started: teacher-dev → gen0-dev → self-play gen1
  (boss, rest) → gen1 fine-tune → gen1-dev (`drive.sh`, job log `results/drive.log`).
- 23:31 ab-teacher-dev playing: 1,000 fights confirmed (dev_fights.csv regenerated: 600 boss, 300 elite, 65 easy,
  34 hard, 1 event). Memory during play ~3.8 GB (the 10 GB peak is setup only).
- Note: twice a file written by the editor tool was not yet visible to the next command. Verify files before
  running anything that depends on a fresh edit.
- 23:45 **ab-teacher-dev done** (15 min): 771 / 1000 wins, mean terminal value 0.385 (stored bootstrap on the same
  fights: 757, 0.370). 00:01 **ab-gen0-dev done** (16 min): 777 / 1000, 0.387. Small v2 search speed is fine.
- **gen0 vs teacher** (`results/gen0-vs-teacher.md`): +0.31 HP-eq/fight (−0.84, +1.43); wins 77.1% → 77.7%.
  Bosses: hexaghost +0.7 (−2.0, +3.5), slime +1.5 (−2.0, +4.9), guardian −0.3 (−3.2, +2.7). Elites: nob +1.0,
  lagavulin −0.3, sentries −1.4 (all CIs cross 0). Per run: −0.2 (−5.3, +4.1); boss +0.5, elites −0.3,
  hallway+events −0.4 (one lost event/gremlin_nob fight, −38 HP-eq, dominates the 100-fight "other" group).
  → gen0 already ≈ teacher (the old v1 act1-gen0 was −1.2/fight behind its teacher).
- analyze.py: hallway + events are now pooled as one natural-mix group for the per-run estimate (they have 1–17
  dev fights per encounter). Bosses and elites stay per encounter.
- Correction: bucket 4 has 971 boss fights (~330 per boss), not 330 total. Boss self-play = 1,942 plays (~45 min).
- 00:42 **ab-selfplay-gen1-boss done** (40 min): 971 boss decks × 2 = 1,942 fights, gen0 won 1,270 (65.4%).
  `ab-selfplay-gen1-rest` running: 4,348 fights (330 per easy/elite encounter; all hard (≤ 251 each) and events),
  ~33 min.
- 01:11 **ab-gen1 fine-tune** (1 min): 110.6k train / 12.4k valid decision rows; best epoch 2 (valid MSE 0.01006),
  later epochs overfit. 01:31 **ab-gen1-dev done** (18 min): 781 / 1000, mean terminal value 0.397.
- 01:31–08:42 machine idle: the orchestrating agent ran out of tokens. gen2 and forever.sh were not run.
- 08:45 analysis: gen1 vs teacher +1.56 HP-eq/fight (+0.33, +2.77); Slime Boss +5.5 (+1.9, +9.1), Guardian +1.4,
  Hexaghost −0.2, Nob +2.3 (+1.1, +3.8). gen1 vs gen0 +1.25 (+0.18, +2.34). Written up in REPORT.md.
