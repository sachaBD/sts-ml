# Runbook (timestamps local BST)

2026-10-05 22:45 — Plan drafted (PLAN.md). Nothing launched. Library v1 finished 400/400
(ac2e6873 all 20 errored: unsupported Scrawl). Win-only diagnostic owned by single-fight-investigation.
2026-10-05 22:50 — User rules: max 10 gameplay workers total at any time; no bulk human pulls (megacrit_dump only small/bounded if needed).
2026-10-05 22:55 — Win-only diagnostic complete (single-fight-investigation): rescue 2/44, both sanity regressed. Win-only teacher lever dropped from plan.
2026-10-05 23:05 — Plan reviewed with single-fight-investigation; adopted: family-level holdout + consumed controls,
teacher50 parity, experience-matched control, provisional graduation, per-deck taper/quotas, mixed-deck minibatches,
RSS caps, 4-seed probe screen, S3 budget (≤12 decks × 30). 3 r01-val library decks training-protected.
Agreement reached (plan only); awaiting user sign-off.
2026-10-05 23:10 — USER APPROVED. Horizon: keep using compute until stopped or ~24 h (to ~2026-10-06 23:00);
results ready for user check-in ~07:00 local. S3-style confirm/report checkpoint by 07:00, then continue S2 after.
orch-26-10-5 and impl-26-10-5 reachable (orch: load 0.1, 11 GiB available).
(Note: entries above labelled 22:45–23:10 were approximate; times below are UTC.)
2026-10-05T21:50Z — splits frozen: apps/corpus/splits.py → runs/schema=combat_v4/date=2026-10-05/id=human-deck-corpus/splits/
  pool 1,028 decks (r01 non-val + r01-unmapped, family-filtered) ; 540 held-out families; 8 consumed controls;
  3 library decks training-protected (de0524bd, 9308c94d, ebad9892).
2026-10-05T21:56Z — LAUNCH S0 screen wave1: id=hdc-screen-w1 (apps/corpus/screen.py), 40 new decks (8/bucket),
  probe 4×MCTS20k then fill 16 for probe-pass decks (all completed, mean ≤60 s). 10 workers. Est 40–75 min.
  Dashboard: .venv/bin/python -m apps.corpus.screen status --run runs/schema=combat_v4/date=2026-10-05/id=hdc-screen-w1
2026-10-05T22:05Z — impl-26-10-5 assigned SPEC_CONTROLLER.md (controller + train.py --mix-shards/--groups). orch-26-10-5 SRE on S0.
2026-10-05T22:16Z — S0 DONE (20 min): 40 decks screened, 0 software/cost failures; 27 in band 1–19/20, 8 at 0/20, 5 at 20/20.
2026-10-05T22:25Z — decks-v1 built (apps/corpus/decks.py → .../id=human-deck-corpus/decks-v1/): 46 eligible
  (incl. library), pilot wave0 = 7a9ada48(A) 168ec20d 8c6f8ad1 1676535e c37ba496 327e6f93 d324d0c5 3cbcfe8a;
  waves 1–5 queued (wave 5 = 20/20 anchors). Skipped: 10×0/20, 478ffcb8 cost, ac2e6873 Scrawl, 3 protected.
  Controller reviewed (impl); smoke pending.
2026-10-05T22:19Z — impl smoke PASSED (scratch/corpus-smoke; entry→bootstrap→eval→collect→train mixed→wave+zero-shot→resume).
2026-10-05T22:19Z — LAUNCH S1 pilot: id=hdc-pilot-joint (apps/corpus/controller.py run, decks-v1 wave0, fresh width64 sigmoid,
  --updates 8, 10 workers, exclude-starts = pool/bench/library/screen/A-run/ref starts). control.json (from k=1):
  50 fights/deck/update fixed (min 50, budget 400 → curriculum OFF in pilot for experience-matched control),
  wave_every 1000 (no new waves), eval_every 4, replay 300/deck, teacher 50/deck taper 5/update.
  Dashboard: .venv/bin/python -m apps.corpus.controller status --run runs/schema=combat_v4/date=2026-10-05/id=hdc-pilot-joint
2026-10-05T22:47Z — pilot k=0 eval (bootstrap, 50 teacher/deck): learned2k monitor wins/30 vs MCTS: A 26 vs 28; 168ec20d 13 vs 16;
  8c6f8ad1 3 vs 12; 1676535e 10 vs 18; c37ba496 4 vs 14; 327e6f93 4 vs 9; d324d0c5 5 vs 12; 3cbcfe8a 9 vs 7.
  Update 1 collection p 6–17% (explore on) except A 73%. Train-1 RSS 2.4 GB, 56 s. ~8–10 min/update.
  Dashboard needs out/: .venv/bin/python -m apps.corpus.controller status --run runs/schema=combat_v4/date=2026-10-05/id=hdc-pilot-joint/out
2026-10-05T23:15Z — pilot k=4 eval: A 30/30 (MCTS 28) → graduated; 168ec 13 (16); 8c6f 3 (12); 1676 9 (18); c37b 4 (14);
  327e 7 (9); d324 8 (12); 3cbc 11 (7). Collection p (explore) non-A mean ~0.11 → ~0.24. Train RSS 3.3 GB.
2026-10-06T00:01Z — S1 pilot DONE (1.71 h). LAUNCH control id=combat_v4/2026-10-06/hdc-pilot-control-168ec20d (seeds verified
  identical to joint for 168ec20d teacher/monitor). Note: runs.run compacted pilot out/*.parquet (starts files) into
  compact-*.parquet at exit — harmless (controller regenerates), but future S2 continuation runs the controller directly.
2026-10-06T00:10Z — S2 prep: controller patched (taper_after_mastery; decks file may change on resume if entered decks kept;
  decks history logged). Pilot config.json: removed decks_sha (backup config.pilot.json); state backup state.pilot-k8.json;
  mastered_at retro-filled. control.json set for S2. Unit tests 13 OK.
2026-10-06T00:17Z — control DONE: 168ec20d separate 7/30, 8/30, 7/30 at k=0/4/8 vs joint 13, 13, 18 (MCTS 16).
  Paired k=8 same seeds: joint 18 vs separate 7; discordant 13 vs 2 (exact McNemar p≈0.007). n=1 deck, experience-
  matched (joint also has 8× more total gradient data). Decision per rule: scale JOINT.
2026-10-06T00:18Z — LAUNCH S2 in place (pilot out dir), controller directly (command in s2-command.sh, pid in out/s2.pid,
  log out/logs/s2-launch.log + controller.log), decks-v2-easyfirst, --updates 1000 (until stop), control: 480/update,
  curriculum on, wave every 3, eval every 4, replay 200/deck, taper_after_mastery.
2026-10-06T00:27Z — S2 CRASH in enter-9 encode: deck 6fe8f9ef has RAGNAROK, unsupported by the PV encoder (MCTS plays it).
  orch reported 00:45Z; I was mid-sleep → ~45 min idle (lesson: sleep ≤25 min).
2026-10-06T01:10Z — encodability pre-check of all queued decks (1 screen fight each through pv.data): 2 bad —
  6fe8f9ef (RAGNAROK), abe6e6de (ALPHA). Removed (software scope) → decks-v3-easyfirst-encodable.json (44 decks);
  pending k=9 plan edited to drop 6fe8f9ef (state backup state.pre-fix-k9.json). Wave 1 = 7 decks.
2026-10-06T01:11Z — S2 RESUMED (s2-command.sh updated). enter-9 MCTS was already done; encodes + zero-shot running.
2026-10-06T01:45Z — wave 1 (7 easy decks, MCTS 18–30/30 on monitor) entered k=9. Zero-shot learned2k (k=8 model, no training on
  them): 104/210 (50%) vs MCTS 180/210 (86%) — transfer present, far below MCTS. ~9.7 min/update, RSS 4.2 GB.
  impl hardening (entry encode failure → software-failed; decks.py --check-encodable) done; applies at next restart.
2026-10-06T02:12Z — wave 2 entered k=12 (23 decks). control: wave_every 3→6 (decks were entering after only ~70–170
  learner fights; A needed ~500), fights_per_update 480→640 (amortize training). Next wave at k=18.
2026-10-06T02:40Z — eval k=12 (monitor wins/30, MCTS): wave1 easy decks sum 142/210 (zero-shot 104) vs MCTS 180; 00205aa1 26 vs 24.
  Original 7 hard decks 60/210 (k=8: 65) — flat; MCTS 87. A 30/30. Wave 2 (entered k=12, after 1 update of training) sum 73/238 vs MCTS 139.
  RSS 5.2 GB. control: wave_every → 1000 (hold wave 3 until after S3 so compute goes to graduations before 07:00 local report).
  Plan: S3 after eval k=20 (~05:00Z): stop → freeze → confirm.py → resume.
2026-10-06T03:36Z — cf0e8f0b (block, 56/75) → software-failed by rule: 2/17 learner fights 'capped' at k=13 (fight-length cap;
  likely learned defensive stalling, not a native error). Kept excluded per pre-registered rule; reported separately.
2026-10-06T04:00Z — control stop=true: S2 pauses after update 16 (incl. eval-16); then freeze + S3 confirm, then resume S2.
TIME CORRECTION: controller.log uses local BST; entries above labelled 03:36Z/04:00Z were really ~02:36Z/~03:00Z.
2026-10-06T03:33Z (true UTC) — S2 paused cleanly after k=16. eval-16 (wins/30 vs MCTS): A 30 (28) grad; 84744719 27 (27) grad;
  wave1 easy sum 155/210 (k=12 142, zero-shot 104) vs MCTS 180; 00205aa1 25 vs 24, 99c688dd 18 vs 18.
  Wave2 sum (7 live) 85/210 vs MCTS 123. Original hard 7: 56/210 (k=8 65, k=12 60) vs MCTS 87 — slowly DECLINING
  (c37ba496 7→2→1). RSS 6.4 GB.
2026-10-06T03:35Z — S2 RESUMED (stop=false) for k=17–20 (eval at 20, ~04:35Z), then S3 (final seeds consumed once, at k=20).
2026-10-06T04:38Z — MACHINE RESTART (unknown cause; user suspects Windows update) during eval-20 (465/~690 monitor fights
  persisted). k=20 model trained and saved (models/20). S3 NOT run. Idle until 07:11Z. Nothing running now.
  Resume: `bash experiments/human-deck-corpus/s2-command.sh` (stop=true is set → it finishes eval-20, then exits), then S3.
2026-10-06T07:30Z — Handed over to user (HANDOFF.md). No processes running.
