# Log (local time, BST)

## 17:15 — start (plan: PLAN.md)
- Committed the staged combat_v4 recorder/decompressor + act3-heart work (5b62883).
- PV play throughput (random-init net, Champ fixture fight, 1 core): w64 ≈ 20k sims/s (35 decisions × 5000 sims in
  8.6 s); w128 ≈ half. So ~5k sims/decision fits the ~14 s/fight budget. Leaves are cheap; no rollouts needed.
- Bug: `pv_worker play` "selected action is not legal" (CombatEnvironment enumerator vs pv::legal_actions pick
  different representatives of identical cards). Fix delegated (impl T1).
- impl T1: Parquet feature cache, Parquet-reading trainer (GPU), play runner writing combat_v4 tables (PV + teacher
  agents) with depth telemetry, CTest.
- Bench: all 453 Champ fights from the fresh-seed runs (981e9+; ah-fresh-inc/-r0b-v1td/-r0d-v32d decks; never
  trained on). Recorded teacher result: 40.8% win, mean HP-eq score 26.4, start HP 60.9.
- `apps/pv/starts.py`: bench selection and the self-play deck generator (real Act ≥ 2 starts moved to the Champ room,
  fresh battle seed; add 1–2 scaling cards p .3, strip scaling p .1, HP redraw p .3).

## 17:40 — teacher Champ baseline (experiments/combat-pv/champ_diag.py on combat_v4_full ah-champ, n=1955)
| | n | win | crossed 50% | turn of cross | player Str at cross | HP at cross | turns cross→end | Execute turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 1955 | 42% | 85% | 7.3 | 7.1 | 31 | 1.9 | 0.54 |
| Demon Form | 438 | 81% | 90% | 7.1 | 17.9 | 34 | 1.2 | 0.29 |
| no DF | 1517 | 31% | 83% | 7.3 | 3.7 | 30 | 2.1 | 0.62 |
- The teacher crosses late (turn ~7) having already lost about half its HP; no-DF decks then need ~2 more turns and
  win 37% of the time once crossed. 15% of fights end before Champ is ever below 50%.

## 00:05 (Oct 4) — machine died (OOM), ~6 h lost
- Cause (impl's account + kernel log): a background `data.py --parts 8` encode over all ah-* combat data, 8 python
  processes each running an unbounded DuckDB join/group-by of 2.4M search rows (DuckDB default memory limit ≈ 80%
  of RAM per process). OOM kills at 18:20, 19:26, 21:02, 22:24 (8 × ~2 GB RSS), box dead at 22:40, rebooted 00:02.
  Other sessions also had the load at ~126.
- Fixes: no fan-out; DuckDB jobs set memory_limit/threads; Champ input pre-filtered once to
  runs/schema=combat_v4/date=2026-10-04/id=champ-ah (1955 fights, 72k search rows, 2.7 MB). Every job > 1 min goes
  through me with a memory bound.

## 00:50 — bootstrap from the teacher: weak (as expected), so straight to self-play
- Bench = 409 fresh-seed Champ starts without Runic Dome (`champ-bench-nodome`; 44 of 453 have Dome, which PV
  search rejects: hidden-intent beliefs not implemented — known gap, to fix before Act 2 bosses at large).
  The teacher is deterministic per start: a 20k rerun reproduced the recorded result on every fight
  (185/453; nodome 41.8%), 13.4 s/fight on 8 workers.
- Teacher visit targets are nearly flat: mean entropy 1.37 nats over 5.2 legal moves, max visit share 0.36
  (UCB1 + rollouts spreads visits). Policy CE barely moves (1.53 → 1.49 val); val top-1 vs the teacher's choice 0.39;
  sharpening (p ∝ visits^4) doesn't help (0.38–0.41).
- Value overfits fast: 1.3k training fights; train value loss 0.054 → 0.002 by epoch 20, val best at epoch 1
  (0.057 ≈ 24 HP-pt RMSE). Data-limited.
- PV-boot (w64, epoch 1) at 2000 sims on the bench: **25.9% vs teacher 41.8%** (paired b/c 72/7, score −9.6 ± 1.2),
  6.8 s/fight; search depth mean 4.2 actions / 1.0 turn, p90 max 11 actions / 4 turns.
- Next: self-play loop (selfplay.py): 3000 generated starts/round, 800 sims + root noise, train from the previous
  checkpoint on the last 3 rounds, bench at 2000 sims. ~25 min/round.
