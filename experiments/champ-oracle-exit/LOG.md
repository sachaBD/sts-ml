# champ-oracle-exit log (UTC)

## 2026-10-04 ~10:10 — code e902f9a; iteration 0
- Iteration 0: fresh net width 64, 4502 teacher fights (champ-train + teachergen-a), value = 100·won (MSE/100²),
  policy = teacher visits. Best epoch 0 (val value 0.111 ≈ RMSE 0.33 in win prob; policy CE 1.540, top-1 0.329);
  value overfits from epoch 1 (train 0.025 / val 0.128) — per-fight shared labels, 4.5k fights.
- Smoke (first 20 bench starts): plumbing OK, replays OK. Oracle800 1/20, real2000 3/20, teacher 6/20.

## 10:32 — sweep0 (iteration-0 net, 409 nodome bench, teacher 41.8%)
| arm | win | DF / no-DF | s/fight | turns/sim mean / p90 |
|---|---:|---|---:|---|
| policy-only | 20.3% | 54.0 / 11.2 | 0.01 | – |
| oracle 200 | 23.0% | 66.7 / 11.2 | 0.75 | 0.99 / 2 |
| oracle 800 | 25.9% | 60.9 / 16.5 | 3.06 | 1.28 / 2 |
| real 2000 (8 particles) | 21.8% | 57.5 / 12.1 | 8.22 | 1.00 / 2 |
| oracle 3200 | 27.6% | 72.4 / 15.5 | 14.51 | 1.60 / 3 |
95% CI ≈ ±4 pp per arm. All p < 0.001 vs teacher.
- Oracle > real (+4 pp at 800), so no sign of a reuse bug. Search adds little over the raw policy: the net is the
  bottleneck, as expected for a weak warm start. Depth ~1–1.6 turns.
- S = 800 by the pre-registered rule (smallest within 2 pp of best). N = 8000 (≈45 min self-play on 9 workers).
- Launched exit.py tag a: 3 rounds, N 8000, S 800, 2 epochs, window 3; review after round 3.
- Null-result criterion for this block: real2000 bench not above iteration-0 21.8% by > 1 SE (≈2 pp) after 3 rounds,
  or val value loss not improving on fresh data → stop and discuss with the user.

## ~10:45 — round size 3000, value target mixed
- tag a (8000/round) and tag b (3000/round, outcome-only value) stopped early in round 1 by decision, not failure.
- Change 1 (b0fd7ef): 3000 fights/round (~20 min self-play) for faster feedback; real2000 bench every 3rd round.
- Change 2 (f5e7299): value target = 0.5·outcome + 0.5·oracle search root value (both 100·win-prob). Rationale:
  outcome labels are shared by ~40 states/fight and memorized at our scale (iteration-0 val minimum at epoch 0);
  root values are lower-variance and, in oracle mode, chance-free; prior session's only gain came from search-value
  distillation (23% → 30.8%). Note val value loss is now vs the mixed target (not comparable to 0.111).
- Launch tag c: 9 rounds, N 3000, S 800, 1 epoch, window 3, value-mix 0.5, real-every 3 (~4.1 h). Rationale for N in PLAN.md.
- Review points: round 3 (first real bench) and round 6. Null at round 3/6: real2000 ≤ ~24% (iteration-0 21.8% + 1 SE).
