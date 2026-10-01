# elite-v3 execution log

Append entries as `- <UTC time> <stage>: <what> (run id, status, wall time, key numbers, gate decision)`.

## Environment
- sts_combat_rl HEAD: f44b3ecfb5fcd68e903fe7931806867da10d5f1d (dirty working tree, owner's uncommitted changes, untouched)
- sts_lightspeed HEAD: 1eef26671f057f7091b835848b256ec3c32c2b2e
- GPU / RAM: RTX 3060 12 GiB (1.7 GiB used at start); 15 GiB RAM (14 available), 4 GiB swap; 12 CPUs
- T0: 2026-09-27T22:37:50Z

## Entries
- 22:37Z stage 0: preflight started (T0). Environment recorded above.
- 22:38Z stage 0: make build OK; ctest 6/6 passed; python unittest 48 passed (incl. test_deep_sets_v3, 9/9). Gate: pass.
- 22:40Z stage 0: smoke elite-v3-smoke (scratch) done: 6/6 fights, 6 wins, skipped_diverged 0/6, 0.52 s/fight; check_data GATE PASSED (421 rows, 115 decisions, potions on 3/6 fights, 6 relic ids).
- 22:40Z stage 1: launched elite-v3-teacher (s1-teacher.toml, 11 workers, 20k/8). Expected ~95 min.
- 23:22Z stage 1: elite-v3-teacher done (35.3 min wall, 4.73 s/fight/worker): 4,884 fights of 5,100 sources, skipped_diverged 216 (4.2%), wins 4,390, mean terminal value 0.458. check_data **GATE PASSED (G1)**:
    parts 4884, rows 377,175, decision rows 92,576, fights 4,884
    fights per encounter: {'three_sentries': 1635, 'gremlin_nob': 1632, 'lagavulin': 1617}
    wins 4390 / 4884
    skipped (replay diverged): 216 of 5100 source fights; teacher {'child_min_visits': 50, 'chunk': 500, 'early_stop': True, 'forced_simulations': 500, 'leaf': 'guided_rollout', 'max_actions': 512, 'oracle': False, 'particles': 8, 'random_move': False, 'random_window': 24, 'simulations': 20000}
    legal moves per decision: mean 4.2, max 512; tried moves mean 4.2
    fights starting with >=1 potion: 2730 / 4884
    potions at fight start: {'Swift': 136, 'Energy': 132, 'Speed': 131, 'Fear': 127, 'Blessing of the Forge': 125, 'Strength': 124, 'Block': 123, 'Power': 123, 'Colorless': 121, 'Dexterity': 120, 'Weak': 119, 'Explosive': 116, 'Fire': 114, 'Blood': 109, 'Flex': 109, 'Skill': 107, 'Distilled Chaos': 103, 'Duplication': 98, 'Ancient': 94, 'Attack': 94, 'Essence of Steel': 91, 'Regen': 89, 'Liquid Memories': 87, 'Liquid Bronze': 86, 'Elixir': 78, "Gambler's Brew": 72, 'Fairy': 59, 'Fruit Juice': 54, 'Snecko Oil': 51, 'Entropic Brew': 49, 'Heart of Iron': 44, 'Cultist': 42}
    relic ids at fight start (RelicId): 64 distinct; most common [(86, 4884), (89, 107), (26, 106), (76, 100), (121, 99), (45, 99), (44, 96), (46, 95), (48, 95), (119, 95), (9, 94), (4, 93)]
    GATE PASSED
    Note: legal moves per decision max 512 (= max_actions cap); mean 4.2.
- 23:22Z stage 2: launched elite-v3-t1 (then elite-v3-t2 chained after it) on GPU, and elite-v3-v-mcts (10 workers) on CPU in parallel. Elapsed T0+0:44.
- 23:35Z stage 2: CORRECTION to the previous entry: t2 was not chained; t1 and v-mcts were launched together. **elite-v3-t1 first launch was OOM-killed (exit 137, kernel oom-kill) in the launcher's `--inputs` pre-step, before any run directory was created.** Cause: both launchers ran their `--inputs` query at once; value_play's alone peaked at ~8.6 GB RSS on 15 GB RAM. This is not the trainer running out of memory, so the t1d fallback does not apply yet. No elite-v3-t1 run was registered, so the id is not reused in the runs/ sense; relaunching as elite-v3-t1 alone (templates depend on that id).
- 23:35Z stage 2: elite-v3-v-mcts done (3.4 min wall, 4.6 s/fight/worker): 421 fights of 450, skipped_diverged 29, wins 376/421, mean terminal value 0.452 (stored teacher on the same fights: 367 wins, 0.436).
- 23:35Z stage 2: relaunched elite-v3-t1 (alone). Elapsed T0+0:58.
- 23:41Z stage 2: elite-v3-t1 done (~5 min; 10 epochs × ~25 s; 4,406 train / 478 validation episodes; RSS ~4 GB). Best epoch 10 (keep=best). validation_mse 0.007575 (teacher_mse 0.006851, baseline 0.039065). Final v3 line: train_aux_keep_mse=0.00219 train_policy_ce=0.92605 policy_validation_ce=0.94338 policy_validation_top1=0.70139 policy_validation_states=9437 aux_keep_validation_mse=0.15176. Note: aux_keep validation MSE (0.15) is about 70× its train MSE and rose over training; flag for the report.
- 23:41Z stage 2: launched elite-v3-t2.
- 23:47Z stage 2: elite-v3-t2 done (~5 min). Best epoch 6 (keep=best), validation_mse 0.007155. Best-epoch v3 line: train_aux_keep_mse=0.01132 train_policy_ce=0.92334 policy_validation_ce=0.93904 policy_validation_top1=0.70934 aux_keep_validation_mse=0.17125. Final (epoch 10) v3 line: train_aux_keep_mse=0.00327 train_policy_ce=0.90382 policy_validation_ce=0.93646 policy_validation_top1=0.70987 policy_validation_states=9437 aux_keep_validation_mse=0.16440. (Trainer prints each epoch's v3 line just before that epoch's summary line.)
- 23:47Z stage 3: rendered s3-v-t1-value / s3-v-t1-policy / s3-v-t2-policy; running them sequentially. Elapsed T0+1:12.
- 23:58Z stage 3: elite-v3-v-t1-value done (8.5 min wall, 12.8 s/fight/worker): 421 fights, skipped_diverged 29, 365 wins, mean TV 0.440.
- 00:08Z stage 3: elite-v3-v-t1-policy done (7.1 min, 10.7 s/fight): 421 fights, skipped 29, 373 wins, mean TV 0.454.
- 00:17Z stage 3: elite-v3-v-t2-policy done (6.9 min, 10.3 s/fight): 421 fights, skipped 29, 378 wins, mean TV 0.458. (Its run dir is date=2026-09-28: the UTC date rolled over.)
- 00:18Z stage 3: compare → results/stage3a.md/.json. Overall HP-eq vs MCTS: t1-value −1.46 [−2.78, −0.11]; t1-policy +0.27 [−0.85, +1.42]; t2-policy +0.82 [−0.39, +2.07]. 421 pairs, 0 unpaired.
- 00:18Z **Gate G3a: best = elite-v3-t2** (policy arm +0.82 > t1 +0.27); policy weight 0.25.
- 00:18Z stage 3: rendered the sweep with best=elite-v3-t2; run ids come out as elite-v3-v-elite-v3-t2-c05 / -c2 (the template's naming). Running sequentially.
- 00:27Z stage 3: elite-v3-v-elite-v3-t2-c05 done (6.0 min, 9.0 s/fight): 421 fights, skipped 29, 379 wins, mean TV 0.458.
- 00:37Z stage 3: elite-v3-v-elite-v3-t2-c2 done (7.7 min, 11.6 s/fight): 421 fights, skipped 29, 372 wins, mean TV 0.451.
- 00:44Z stage 3: compare → results/stage3b.md/.json. Overall HP-eq vs MCTS: c0.5 +0.82 [−0.37, +2.03]; c2.0 −0.01 [−1.20, +1.16]; c1.0 (stage 3a t2-policy) +0.82 [−0.39, +2.07].
- 00:44Z **Gate G3b: c_puct = 1.0 (tie rule).** Exact JSON values: c0.5 = 0.8194776824, c1.0 = 0.8194776561. The gap (2.6e-8 HP-eq/fight) is float32 rounding in terminal_value. The smallest real outcome difference (1 HP in one fight) moves the mean by ~0.0024, so this counts as a tie → 1.0. Per elite the arms differ (c0.5: Nob +1.41, Lag +0.66, Sent +0.39; c1.0: +1.50, −0.01, +0.94).
- 00:45Z **Gate G4: pass** (elapsed T0+2:07 ≤ 4:30). Rendered s4-selfplay (best=elite-v3-t2, c_puct=1.0); launched elite-v3-selfplay.
- 01:09Z stage 4: elite-v3-selfplay spent ~24 min (00:45–01:09Z) in the launcher/generate.py query phase before playing any fight. The DuckDB scan of the combat_v3 view sat at its default memory limit (~14 GB RSS on 15 GB, swap full, spilling). The view now covers the teacher's 4,884 parquet parts plus the validation runs. It recovered without intervention; fights started at 01:09Z. Risk for later stages: value_play / train launches may pay the same cost.
- 01:49Z stage 4: elite-v3-selfplay done (38.0 min of play after the ~24 min query phase; 10.68 s/fight/worker): 2,335 fights of 2,436 sources, skipped_diverged 101 (4.1%), wins 2,064, mean TV 0.460. check_data **GATE PASSED**:
    parts 2335, rows 166,797, decision rows 45,668, fights 2,335
    fights per encounter: {'lagavulin': 783, 'three_sentries': 842, 'gremlin_nob': 710}
    wins 2064 / 2335
    skipped (replay diverged): 101 of 2436 source fights; teacher {'batch': 64, 'c_puct': 1.0, 'child_min_visits': 50, 'early_stop': True, 'forced_simulations': 500, 'fpu_reduction': 0.05, 'leaf': 'policy_net', 'max_actions': 512, 'objective': 'won * (35 + hp + 4 * potions) / (56 + root max hp)', 'oracle': False, 'particles': 8, 'prior_floor': 0.03, 'random_move': False, 'random_window': 24, 'simulations': 20000}
    legal moves per decision: mean 4.2, max 128; tried moves mean 4.2
    fights starting with >=1 potion: 1325 / 2335
    potions at fight start: {'Attack': 80, 'Colorless': 71, 'Weak': 69, 'Fear': 68, 'Strength': 64, 'Flex': 64, 'Explosive': 62, 'Power': 61, 'Energy': 60, 'Blessing of the Forge': 59, 'Speed': 58, 'Skill': 56, 'Swift': 55, 'Dexterity': 55, 'Fire': 54, 'Blood': 53, 'Block': 52, 'Elixir': 43, 'Liquid Memories': 42, 'Distilled Chaos': 41, 'Ancient': 38, 'Essence of Steel': 36, 'Liquid Bronze': 35, 'Duplication': 34, "Gambler's Brew": 34, 'Regen': 32, 'Fairy': 25, 'Snecko Oil': 24, 'Cultist': 23, 'Heart of Iron': 23, 'Fruit Juice': 21, 'Entropic Brew': 21}
    relic ids at fight start (RelicId): 63 distinct; most common [(86, 2335), (45, 65), (52, 53), (89, 53), (55, 46), (91, 46), (90, 45), (70, 45), (41, 44), (119, 44), (33, 44), (97, 43)]
    GATE PASSED
- 01:49Z stage 4: rendered s4-t3 (best=elite-v3-t2, policy_weight=0.25); launched elite-v3-t3.
- 01:55Z stage 4: elite-v3-t3 done (~6 min; 6 epochs × ~36 s). Best epoch 1 (keep=best), validation_mse 0.004845 (teacher_mse 0.005537, baseline 0.034913). Best-epoch v3 line: train_aux_keep_mse=0.04126 train_policy_ce=0.88164 policy_validation_ce=0.86911 policy_validation_top1=0.74779 policy_validation_states=13453 aux_keep_validation_mse=0.05469. Final (epoch 6): policy_validation_ce=0.86586 top1=0.75203 aux_keep_validation_mse=0.06118. Caveat: split="fresh" over teacher+self-play, and the initial checkpoint (t2) trained on teacher episodes, so t3's validation set partly overlaps t2's training data. Its validation MSE/CE is optimistic and not comparable with t1/t2; best epoch 1 also means later epochs did not improve on this split.
- 01:55Z stage 4: rendered s4-v-t3-policy (c_puct=1.0); launched elite-v3-v-t3-policy.
- 02:05Z stage 4: elite-v3-v-t3-policy done (6.6 min, 9.9 s/fight): 421 fights, skipped 29, 370 wins, mean TV 0.453. compare → results/stage4.md/.json: overall +0.21 [−0.96, +1.39]; Nob +1.42, Lag +0.57, Sentries −1.35.
- 02:05Z stage 5: **candidate = elite-v3-t2, policy arm, c_puct 1.0** (validation run elite-v3-v-t2-policy). Overall HP-eq by arm: t2-c1.0 +0.8194776561, t2-c0.5 +0.8194776824, t1-policy +0.27, t3-policy +0.21, t2-c2.0 −0.01, t1-value −1.46. The top two tie within float32 precision (see G3b); as in G3b, the tie resolves to c_puct 1.0. Judgment call: the plan has no explicit tie rule for the candidate. The c0.5 arm would also have passed G5 (per-elite +1.41/+0.66/+0.39).
- 02:05Z **Gate G5: PASS.** (a) elapsed T0+3:28 ≤ 6:45; (b) overall +0.82 ≥ 0; (c) per elite Nob +1.50, Lag −0.01, Sentries +0.94, none < −1.0. Launching elite-v3-f-mcts, then elite-v3-f-candidate.
- 02:19Z stage 5: elite-v3-f-mcts done (10.2 min of play, 4.6 s/fight/worker): 1,410 fights of 1,500, skipped_diverged 90, wins 1,289, mean TV 0.464.
- 02:45Z stage 5: elite-v3-f-candidate (t2 + policy priors, c_puct 1.0) done (22.0 min of play, 10.0 s/fight/worker): 1,410 fights, skipped 90, wins 1,282, mean TV 0.467.
- 02:46Z stage 5: compare → results/final.md/.json. **Overall +0.42 [−0.16, +0.98] HP-eq/fight** (1,410 pairs, 0 unpaired); Nob +1.00 [+0.31, +1.70], Lagavulin +0.98 [−0.09, +2.03], Sentries −0.72 [−1.85, +0.38]. Wins 1,289 (MCTS) vs 1,282 (candidate); candidate-only / MCTS-only wins 23 / 30.
- 02:46Z stage 6: per-run summary. Wall = run.json started→finished (includes the launcher/query phase); s/fight is per worker, from summary.json.

| run id | status | fights | skipped (diverged) | wall min | s/fight |
|---|---|---:|---:|---:|---:|
| elite-v3-smoke | done | 6 | 0 | 1.6 | 0.52 |
| elite-v3-teacher | done | 4884 | 216 | 39.5 | 4.73 |
| elite-v3-v-mcts | done | 421 | 29 | 5.1 | 4.63 |
| elite-v3-v-t1-value | done | 421 | 29 | 10.5 | 12.77 |
| elite-v3-v-t1-policy | done | 421 | 29 | 8.8 | 10.69 |
| elite-v3-v-t2-policy | done | 421 | 29 | 8.7 | 10.33 |
| elite-v3-v-elite-v3-t2-c05 | done | 421 | 29 | 7.8 | 9.04 |
| elite-v3-v-elite-v3-t2-c2 | done | 421 | 29 | 9.6 | 11.63 |
| elite-v3-selfplay | done | 2335 | 101 | 56.4 | 10.68 |
| elite-v3-v-t3-policy | done | 421 | 29 | 8.6 | 9.93 |
| elite-v3-f-mcts | done | 1410 | 90 | 13.0 | 4.63 |
| elite-v3-f-candidate | done | 1410 | 90 | 25.4 | 10.02 |
| elite-v3-t1 | done | - | - | 5.4 | - |
| elite-v3-t2 | done | - | - | 5.4 | - |
| elite-v3-t3 | done | - | - | 5.2 | - |
  (Trainings: t1 validation_mse 0.007575 at epoch 10; t2 0.007155 at epoch 6; t3 0.004845 at epoch 1, not comparable. See the entries above.)
- 02:46Z stage 6: results/ contains stage3a, stage3b, stage4, final (.md + .json). Messaged nn-consultant "elite-v3 done". Total elapsed T0+4:08:57.
