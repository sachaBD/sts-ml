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

## 17:15 — end-of-turn state census (impl, c1d687d; out: id=champ-ox-turns-c-r09/out)
- 441 turns / 50 r09 oracle bench fights. Median / p90 / max: sequences 326 / 8.2k / ≥1M; exact distinct
  non-terminal end states 206 / 2.7k / 673k; canonical (sorted hand/discard/exhaust) 66 / 819 / 168k;
  enumeration 4 ms / 95 ms / 15.6 s. 5 turns (1.1%) hit the 1e6 sequence cap.
- Canonical buckets are lossy: 67/200 merged pairs re-diverge after one more turn (33.5%) → use exact keys.
- Implication: turn-level oracle search is affordable for typical turns (≈200 children), needs a tail cap / fallback.
  Network cost and multi-turn search cost not yet measured.
| r10 | selfplay 1821/6000=0.303 cap0 43.6min | e0 val v0.067 ce1.461 top10.519 (train v0.021) | bench-policy 95/409=0.232 DF 53/87 noDF 42/322 0.01s/f cap0 | bench-oracle 180/409=0.440 DF 75/87 noDF 105/322 4.85s/f cap0 turns 1.28/p90 2 |
| r11 | selfplay 1870/6000=0.312 cap0 45.9min | e0 val v0.067 ce1.451 top10.527 (train v0.021) | bench-policy 103/409=0.252 DF 53/87 noDF 50/322 0.01s/f cap0 | bench-oracle 185/409=0.452 DF 76/87 noDF 109/322 4.43s/f cap0 turns 1.26/p90 2 |
| r12 | selfplay 1836/6000=0.306 cap0 46.9min | e0 val v0.060 ce1.455 top10.526 (train v0.020) | bench-policy 96/409=0.235 DF 55/87 noDF 41/322 0.01s/f cap0 | bench-oracle 195/409=0.477 DF 76/87 noDF 119/322 3.36s/f cap0 turns 1.27/p90 2 | bench-real 143/409=0.350 DF 70/87 noDF 73/322 10.22s/f cap0 turns 0.96/p90 2 |
| r13 | selfplay 1967/6000=0.328 cap0 46.3min | e0 val v0.056 ce1.447 top10.534 (train v0.020) | bench-policy 97/409=0.237 DF 54/87 noDF 43/322 0.01s/f cap0 | bench-oracle 192/409=0.469 DF 77/87 noDF 115/322 4.84s/f cap0 turns 1.27/p90 2 |
| r14 | selfplay 1910/5998=0.318 cap2 44.6min | e0 val v0.054 ce1.457 top10.530 (train v0.020) | bench-policy 112/409=0.274 DF 62/87 noDF 50/322 0.01s/f cap0 | bench-oracle 186/409=0.455 DF 76/87 noDF 110/322 3.36s/f cap0 turns 1.28/p90 2 |
| r15 | selfplay 1931/6000=0.322 cap0 37.5min | e0 val v0.059 ce1.443 top10.540 (train v433.252) | bench-policy 97/409=0.237 DF 54/87 noDF 43/322 0.01s/f cap0 | bench-oracle 191/409=0.467 DF 78/87 noDF 113/322 3.68s/f cap0 turns 1.26/p90 2 | bench-real 160/409=0.391 DF 72/87 noDF 88/322 9.56s/f cap0 turns 0.96/p90 2 |

## 23:50 — turn search beats per-action in oracle; tag c r15 real 39.1%
- Turnbench (r09 net, oracle, 409, fixed simulator; baseline per-action 800 = 41.8%, 3.95 s/fight, 1.25 turns/sim):
  ts16 40.8% (2.5 s) · ts64 45.0% (4.4 s, p=0.14) · **ts256 47.9% (10.1 s; b/c 23/48, p=0.004)**. Leaf depth
  1.72 → 1.91 → 2.11 turns (tree max 2.2 → 2.4 → 2.7). Win rate rises monotonically with depth/budget.
  Gain concentrated in no-Demon-Form decks (ts256 noDF b/c 20/41, p=0.010; DF flat). ~14% turns fall back.
  → First direct evidence that deeper (turn-level) planning helps on Champ, with the same network.
- Tag c r15: real2000 39.1% (vs r09 +6.6 pp p=0.001; vs r12 +4.1 pp p=0.06); oracle800 46.7%; policy 23.7%.
  ≥ 38% gate → continue to r18. r15 train value loss spiked to 433 (val fine 0.059) → grad clipping for tag d.
- J1 (PIMC screen, r15) started on CPUs 10–11.
| r16 | selfplay 1964/6000=0.327 cap0 35.6min | e0 val v0.054 ce1.449 top10.533 (train v0.020) | bench-policy 107/409=0.262 DF 57/87 noDF 50/322 0.01s/f cap0 | bench-oracle 190/409=0.465 DF 74/87 noDF 116/322 3.20s/f cap0 turns 1.27/p90 2 |
| r17 | selfplay 2057/6000=0.343 cap0 34.4min | e0 val v0.058 ce1.434 top10.543 (train v0.019) | bench-policy 117/409=0.286 DF 63/87 noDF 54/322 0.01s/f cap0 | bench-oracle 194/409=0.474 DF 77/87 noDF 117/322 3.53s/f cap0 turns 1.30/p90 2 |
| r18 | selfplay 2080/6000=0.347 cap0 35.5min | e0 val v0.058 ce1.438 top10.535 (train v0.020) | bench-policy 108/409=0.264 DF 55/87 noDF 53/322 0.01s/f cap0 | bench-oracle 203/409=0.496 DF 78/87 noDF 125/322 3.23s/f cap0 turns 1.27/p90 2 | bench-real 160/409=0.391 DF 71/87 noDF 89/322 8.62s/f cap0 turns 0.95/p90 2 |

## 02:20 — tag c ends at r18; PIMC turn search rejected; tag d launched
- Tag c real2000: r15 39.1% → r18 39.1% (flat). Oracle800 r18 49.6%, policy 26.4%. Plateau ≈ 39% real (teacher 41.8%).
  Oracle−real gap ≈ 10.5 pp.
- J1 PIMC turn search (r15, K4 E16): 31.3% vs per-action real 39.1% (b/c 57/25, p=0.001), 23.4 s/fight. Clearly worse
  (consistent with strategy fusion: each particle plans with its own known draws). Rejected as a real-play agent;
  bench2k PIMC arm skipped by rule.
- J2 tag d launched (init c-r18, turn-search expert E64, targets, grad-clip 1.0, 9 rounds × 3000).
| r01 | selfplay 1095/3000=0.365 cap0 26.6min | e0 val v0.051 ce1.213 top10.539 (train v0.022) | bench-policy 78/409=0.191 DF 40/87 noDF 38/322 0.01s/f cap0 | bench-oracle 186/409=0.455 DF 76/87 noDF 110/322 4.39s/f cap0 turns 0.93/p90 2 |
| r02 | selfplay 1088/3000=0.363 cap0 26.0min | e0 val v0.056 ce1.207 top10.543 (train v0.019) | bench-policy 90/409=0.220 DF 51/87 noDF 39/322 0.01s/f cap0 | bench-oracle 199/409=0.487 DF 78/87 noDF 121/322 4.26s/f cap0 turns 0.91/p90 2 |
| r03 | selfplay 1063/3000=0.354 cap0 26.4min | e0 val v0.053 ce1.198 top10.546 (train v0.019) | bench-policy 92/409=0.225 DF 53/87 noDF 39/322 0.01s/f cap0 | bench-oracle 186/409=0.455 DF 76/87 noDF 110/322 4.06s/f cap0 turns 0.92/p90 2 | bench-real 137/409=0.335 DF 63/87 noDF 74/322 8.06s/f cap0 turns 0.93/p90 2 |

## 04:05 — tag d regressed; stopped at r03; diagnosis
- Tag d (init c-r18 = 39.1% real): r03 real2000 33.5% (b/c 50/27, p=0.012); policy-only 26.4 → 19.1 (r01) → 22.5.
  Turn-search oracle bench 45.5–48.7% (c-r18 per-action oracle 49.6%; c-r18 turn-search baseline being measured).
- Data check (rows): c-r18 per-action policy targets entropy 1.29 (max share 0.45, 0% one-hot); tag d turn targets
  entropy 0.74 (max share 0.67, **32% one-hot** — within-turn decisions along the played plan collapse to one path).
  Root-value bias vs outcome: c-r18 +0.4; d-r01 +0.3; d-r03 +4.4 (max-backup optimism growing).
  Volume: d-r01 trained on 134k rows vs tag c's ~660k-row window (forgetting risk).
- Candidate causes: (P) sharp oracle-plan policy targets; (V) optimistic turn-search values; (N) small fresh window.
- Diagnostic (all init c-r18, 1 epoch, value-mix 0.5, grad-clip 1.0; bench real2000 + policy-only on 409):
  D1 c-r16..18 rows (control) · D2 d-r01..03 full · D3 d rows value-only · D4 d rows policy-only · D5 c-r16..18 + d-r01..03.
  Reading: D3 ok & D4 bad → P; D4 ok & D3 bad → V; D2 bad & D5 ok → N.
| r04 | selfplay 947/3000=0.316 cap0 26.5min | e0 val v0.058 ce1.178 top10.554 (train v0.019) | bench-policy 84/409=0.205 DF 49/87 noDF 35/322 0.01s/f cap0 | bench-oracle 193/409=0.472 DF 75/87 noDF 118/322 4.46s/f cap0 turns 0.99/p90 2 |
| r05 | selfplay 999/3000=0.333 cap0 27.4min | e0 val v0.057 ce1.168 top10.555 (train v0.019) | bench-policy 103/409=0.252 DF 55/87 noDF 48/322 0.01s/f cap0 | bench-oracle 197/409=0.482 DF 79/87 noDF 118/322 4.25s/f cap0 turns 0.93/p90 2 |
| r06 | selfplay 1066/3000=0.355 cap0 26.3min | e0 val v0.055 ce1.165 top10.556 (train v0.019) | bench-policy 83/409=0.203 DF 48/87 noDF 35/322 0.01s/f cap0 | bench-oracle 195/409=0.477 DF 79/87 noDF 116/322 4.35s/f cap0 turns 0.91/p90 2 | bench-real 142/409=0.347 DF 66/87 noDF 76/322 8.47s/f cap0 turns 0.92/p90 1 |
| diag done 2026-10-05T06:31:42Z D1-D5 see message

## 06:35 — diagnostic results (all init c-r18, 1 epoch; 409 bench; c-r18 real 39.1%)
| model | rows | real2000 | b/c vs c-r18, p | policy-only |
|---|---|---:|---|---:|
| D1 | c-r16..18 (control) | 36.7% | 35/25, 0.25 | 25.7% |
| D2 | d-r01..03 | 36.4% | 41/30, 0.24 | 25.2% |
| D3 | d, value loss only | 31.1% | 54/21, <0.001 | 22.2% |
| D4 | d, policy loss only | 27.6% | 65/18, <0.001 | 23.5% |
| D5 | c + d mixed | **41.3%** | 23/32, 0.28 | 27.1% |
- Turn-search rows are NOT intrinsically harmful: D2 ≈ D1 control. Tag d's regression is best explained by loop
  dynamics: a fresh small window (134k rows at r01 vs ~660k) repeatedly retrained → drift/forgetting (cause N).
- D3/D4 are confounded (shared trunk: training one head degrades the other) — design flaw on my part; they don't
  separate P vs V. Not used for conclusions.
- Retrain noise is material: D1 control moved −2.4 pp from the same init. Single-model 409-bench comparisons can't
  resolve ~2 pp → move evaluation to bench2k (2045 fights).
- D5 (mixed per-action + turn-search data) is the best real model so far (41.3%; teacher 41.8% on the 409 bench).
- c-r18 turn-search E64 oracle 48.4% vs per-action oracle 49.6%: the r09-era turn-search edge at E64 is gone for r18.

### Next (≈ 7 h left)
1. bench2k status (9 workers, ~2 h): teacher (frozen t1, 20k), c-r18 real2000, D5 real2000 — paired.
2. Tag e (~4 h): EXIT with turn-search self-play (E64, turn targets) training on [last 3 tag-e rounds + fixed anchor
   c-r16..18 rows] (replicates D5 iteratively), init D5, 1 epoch, value-mix 0.5, grad-clip 1.0, 6 rounds × 3000;
   real2000 on bench2k at r03 and r06. Stop if r03 is below D5 on bench2k (p<0.05) or r06 not above D5 by > 1 SE.
