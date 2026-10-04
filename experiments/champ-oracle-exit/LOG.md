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
tag c (it0: policy 20.3 / oracle800 25.9 / real2000 21.8; val v0.111 ce1.540 top1 0.329)
| r01 | selfplay 555/3000=0.185 cap0 16.3min | e0 val v0.066 ce1.523 top10.425 (train v0.027) | bench-policy 79/409=0.193 DF 47/87 noDF 32/322 0.01s/f cap0 | bench-oracle 118/409=0.289 DF 61/87 noDF 57/322 3.98s/f cap0 turns 1.22/p90 2 |
| r02 | selfplay 614/3000=0.205 cap0 22.5min | e0 val v0.066 ce1.518 top10.438 (train v0.019) | bench-policy 70/409=0.171 DF 43/87 noDF 27/322 0.01s/f cap0 | bench-oracle 126/409=0.308 DF 62/87 noDF 64/322 5.17s/f cap0 turns 1.23/p90 2 |
| r03 | selfplay 682/3000=0.227 cap0 26.7min | e0 val v0.059 ce1.505 top10.457 (train v0.018) | bench-policy 82/409=0.200 DF 56/87 noDF 26/322 0.01s/f cap0 | bench-oracle 134/409=0.328 DF 69/87 noDF 65/322 4.60s/f cap0 turns 1.24/p90 2 | bench-real 119/409=0.291 DF 64/87 noDF 55/322 12.25s/f cap0 turns 0.97/p90 2 |
| r04 | selfplay 774/3000=0.258 cap0 21.6min | e0 val v0.057 ce1.491 top10.479 (train v0.015) | bench-policy 88/409=0.215 DF 53/87 noDF 35/322 0.01s/f cap0 | bench-oracle 149/409=0.364 DF 71/87 noDF 78/322 4.56s/f cap0 turns 1.25/p90 2 |
| r05 | selfplay 816/3000=0.272 cap0 22.1min | e0 val v0.055 ce1.475 top10.498 (train v0.015) | bench-policy 87/409=0.213 DF 53/87 noDF 34/322 0.01s/f cap0 | bench-oracle 160/409=0.391 DF 75/87 noDF 85/322 4.99s/f cap0 turns 1.24/p90 2 |
| r06 | selfplay 818/3000=0.273 cap0 24.4min | e0 val v0.051 ce1.466 top10.503 (train v0.015) | bench-policy 91/409=0.222 DF 56/87 noDF 35/322 0.01s/f cap0 | bench-oracle 159/409=0.389 DF 74/87 noDF 85/322 4.55s/f cap0 turns 1.27/p90 2 | bench-real 123/409=0.301 DF 64/87 noDF 59/322 13.60s/f cap0 turns 0.97/p90 2 |
| r07 | selfplay 874/3000=0.291 cap0 21.9min | e0 val v0.053 ce1.451 top10.511 (train v0.016) | bench-policy 96/409=0.235 DF 54/87 noDF 42/322 0.01s/f cap0 | bench-oracle 164/409=0.401 DF 73/87 noDF 91/322 4.37s/f cap0 turns 1.24/p90 2 |
| r08 | selfplay 876/3000=0.292 cap0 21.6min | e0 val v0.059 ce1.450 top10.516 (train v0.016) | bench-policy 94/409=0.230 DF 56/87 noDF 38/322 0.01s/f cap0 | bench-oracle 176/409=0.430 DF 71/87 noDF 105/322 4.42s/f cap0 turns 1.28/p90 2 |
| r09 | selfplay 916/2999=0.305 cap1 21.6min | e0 val v0.063 ce1.449 top10.521 (train v0.016) | bench-policy 104/409=0.254 DF 56/87 noDF 48/322 0.01s/f cap0 | bench-oracle 169/409=0.413 DF 73/87 noDF 96/322 3.97s/f cap0 turns 1.25/p90 2 | bench-real 133/409=0.325 DF 68/87 noDF 65/322 9.86s/f cap0 turns 0.96/p90 2 |

## 16:00 — tag c complete (9 rounds) — conclusions
- Real2000: 21.8 → 29.1 (r03) → 30.1 (r06) → 32.5% (r09). vs iteration 0 p<0.001; r03→r09 b/c 22/36 p=0.09;
  r06→r09 20/30 p=0.20. Still below teacher 41.8% (b/c 55/17, p<0.001).
- Oracle800 r09 41.3% vs real 32.5% on the same net: paired 17/53, p<0.001 → transfer gap ~9 pp, grew from ~4 pp.
- Policy-only 20.3 → 25.4%. Val value loss min 0.051 (r06) → 0.063 (r09) (partly base-rate: win rate rising).
- **Depth did not grow**: oracle 1.25 turns/sim (p90 2), real 0.96 (p90 2) at every round. The gains are not from
  longer planning. Expert iteration on the per-action PUCT tree improves evaluation, not horizon.
- Final model: champ-ox-c-r09/model.
