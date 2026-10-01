# elite-bench execution log

Author: elite-bench consultant, intercom `subagent-chat-01a0e977` (01a0e977). Executor: orchestrator-28.

- 21:11Z author: prep done. search_salt tweak added (agents/teacher_leaves.{hpp,cpp}, apps/common/app.py,
  apps/common/teacher_request.hpp; test tests/teacher_leaves_test.cpp test_search_salt). ctest 6/6, python 52 tests OK.
  Smoke: salt 0 reproduced 11/11 stored elite-v3-v-mcts fights exactly; salt 3 differs on 7/11 (one death→win).
  Rehearsal of every stage type on 12 fights (scratch, prefix bsmoke, experiments/elite-bench/smoke/) passed;
  analyze.py/pivotal.py verified on it. Rates: MCTS 20k 4.6 s/fight/worker, oracle 20k 5.5, v3 20k 8.0.
  Configs: .venv/bin/python experiments/elite-bench/make_configs.py (already generated in configs/).

## Execution (orchestrator-28)

- 21:12Z **T0. Stage 0 preflight PASS.** `make build` OK; `teacher_leaves_test: ok`. sts_combat_rl HEAD
  16bf702b064835487ac9ff045155cf253ed3ab78 (dirty working tree as expected, incl. search_salt tweak);
  sts_lightspeed HEAD 1eef26671f057f7091b835848b256ec3c32c2b2e (clean). `free -g`: 15 total, 12 free, 14 avail; swap 4. nproc 12.
  Time gates: bench-mcts-50k-* must start before 03:42Z (T0+6h30); stage 6 must start before 04:12Z (T0+7h00).
- 21:13Z launched stages 1-3 + checkpoint A calcs as one chain (logs/chainA.sh → logs/stages.log).
- 21:31Z DONE bench-oracle-20k (18 min wall): 1676 fights, skipped 124, won 1620 (teacher 1490), 6.3 s/fight.
- 21:46Z DONE bench-oracle-50k (14 min): 565 fights (subset), skipped 35, won 546, 15.0 s/fight.
- 22:00Z DONE bench-mcts-20k-s0 (14 min): 1676 fights, skipped 124, won 1531, 4.6 s/fight.
- 22:14Z DONE bench-mcts-20k-s1 (14 min): 1676 fights, skipped 124, won 1528, 4.5 s/fight.
- 23:43Z author: appended Round 2 to PLAN.md (candidates bench-r2-t4/t5; configs/bench-r2-t*.toml, templates/bench-r2t*; render.py; analyze.py section G). Early A/B results summarised there.
- 00:34Z Round 2 received (author msg 23:43Z, read 00:33Z). Launched GPU training chain bench-r2-t4 then bench-r2-t5 (logs/train_chain.sh → logs/train.log), parallel to the CPU chain (bench-v3-20k-s3 running).
- 22:28Z DONE bench-mcts-20k-s2 (14 min): 1676 fights, skipped 124, won 1525, 4.6 s/fight.
- 22:42Z DONE bench-mcts-20k-s3 (14 min): 1676 fights, skipped 124, won 1523, 4.6 s/fight.
- 23:11Z DONE bench-v3-20k-s0 (28 min): 1676 fights, skipped 124, won 1524, 9.7 s/fight.
- 23:40Z DONE bench-v3-20k-s1 (28 min): 1676, skipped 124, won 1520, 9.7 s/fight.
- 00:09Z DONE bench-v3-20k-s2 (29 min): 1676, skipped 124, won 1513, 9.8 s/fight.
- 00:37Z DONE bench-v3-20k-s3 (28 min): 1676, skipped 124, won 1517, 9.7 s/fight.
- 00:37Z **Checkpoint A**: pivotal.py `fair 20k runs used: 8`; 210 pivotal (nob 25, lagavulin 92, sentries 93) → gate PASS.
  analyze.py → results/analysis-A.md OK.
- 00:37Z (correction) first training launch at ~00:34Z failed instantly (script not yet written — executor race); relaunched
  00:38Z: TRAIN START bench-r2-t4 (logs/train.log, logs/bench-r2-t4.log).
- 00:38Z launched chain B1 (logs/chainB1.sh): stage 5 → stage 7 → R2 evals (each waits for its net; render.py) →
  analysis-R2.md, then auto-starts chain B2 (logs/chainB2.sh): stage 4 → stage 6 with revised gates
  (50k < 05:12Z, S6 < 05:42Z, nothing new ≥ 06:27Z) → analysis.md. CPU idle ~1 min.
- 00:48Z TRAIN DONE bench-r2-t4 (10 min): best epoch 4/10, validation_mse 0.006691. Final v3: line: train_aux_keep_mse=0.00402
  train_policy_ce=0.89862 policy_validation_ce=0.92901 policy_validation_top1=0.71869 policy_validation_states=12950 aux_keep_validation_mse=0.12373.
- 01:03Z TRAIN DONE bench-r2-t5 (15 min): best epoch 2/10, validation_mse 0.005945. Final v3: line: train_aux_keep_mse=0.00513
  train_policy_ce=0.86038 policy_validation_ce=0.89511 policy_validation_top1=0.73968 policy_validation_states=17797 aux_keep_validation_mse=0.13468.
  (Validation splits differ between t4 and t5 because the data differs, so the MSEs aren't directly comparable.) No memory issues.
- 02:41Z author: appended Round 2b (t5 pivotal salts 4-7, then t5 salts 2-3; after stage 6).
- 03:06Z Round 2b received (author msg 02:41Z). Appended to chainB2.sh after stage 6: render+run bench-r2t5-20k-s4..s7, then -s2, -s3, each under the 06:27Z gate; then checkpoint B analysis.
- Stage 5 (00:38–01:10Z), pivotal 210 fights, skipped 0 each: mcts-20k-s4 won 123, v3-s4 102, mcts-s5 121, v3-s5 110,
  mcts-s6 118, v3-s6 110, mcts-s7 124, v3-s7 107 (MCTS ~2 min/run, v3 ~5 min/run).
- Stage 7 (01:10–01:20Z), pivotal 210, skipped 0: mcts-20k-p32-s0 won 122, -s1 118, -s2 121, -s3 121 (~2 min/run).
- R2 evals (1676 fights, skipped 124 each): 01:49Z bench-r2t4-20k-s0 won 1513 (29 min); 02:17Z bench-r2t5-20k-s0 won 1526 (28 min);
  02:46Z bench-r2t4-20k-s1 won 1528 (29 min); 03:15Z bench-r2t5-20k-s1 won 1534 (28 min).
- 03:15Z analysis-R2.md written; chain B2 (stage 4 → 6 → 2b → analysis.md) started automatically.
- 03:32Z author: appended Round 2c (pre-registered confirmation of t5 vs MCTS on 1,262 fresh fights).
- Stage 4 (03:15–04:41Z; 1676 fights, skipped 124 each), won: mcts-1k-s0 1507, -s1 1506; 2k-s0 1507, -s1 1513;
  10k-s0 1520, -s1 1524; 50k-s0 1528 (started 03:40Z, before the 05:12Z gate), -s1 1528. s/fight 0.4 / 0.7 / 2.7 / 11.3.
- Stage 6 (04:41Z–; subset 565 fights, skipped 35 each; started before the 05:42Z gate), won: mcts-5k-s0 513, -s1 513;
  v3-5k-s0 507, -s1 508; v3-50k-s0 512 (21 min, 23.5 s/fight); v3-50k-s1 started 05:12Z.
- 05:15Z Round 2c read (author msg 03:32Z; the executor slept through it on a long wait, so it was read late). To rewire without editing a
  running script: killed the chainB2 bash parent (pid 1003969) between its runs, while the orphaned run.sh bench-v3-50k-s1
  keeps running and logging. Started chainC.sh (logs/chainC.sh): waits for v3-50k-s1 → 2b step 1 (t5 s4–s7, rendered) →
  bench-confirm-mcts-20k-s0 → render + bench-confirm-r2t5-20k-s0 → compare.py → results/confirm.md → bench-r2t5-20k-s2 if before
  06:27Z → analysis.md. 2b's t5 s3 dropped (replaced by 2c).

- 06:52Z **Checkpoint B**: analyze.py → results/analysis.md OK. Round 2c confirm.md: t5 − MCTS all +0.83 [+0.13, +1.55] (n=1163), per elite nob +1.92, lag +0.76, sentries −0.24 → pre-registered criterion MET.

### Per-run summary (id, fights, skipped, wall min)

- bench-oracle-20k, 1676, 124, 18
- bench-oracle-50k, 565, 35, 14
- bench-mcts-20k-s0, 1676, 124, 14
- bench-mcts-20k-s1, 1676, 124, 14
- bench-mcts-20k-s2, 1676, 124, 14
- bench-mcts-20k-s3, 1676, 124, 14
- bench-v3-20k-s0, 1676, 124, 28
- bench-v3-20k-s1, 1676, 124, 28
- bench-v3-20k-s2, 1676, 124, 29
- bench-v3-20k-s3, 1676, 124, 28
- bench-mcts-20k-s4, 210, 0, 2
- bench-v3-20k-s4, 210, 0, 5
- bench-mcts-20k-s5, 210, 0, 2
- bench-v3-20k-s5, 210, 0, 5
- bench-mcts-20k-s6, 210, 0, 2
- bench-v3-20k-s6, 210, 0, 5
- bench-mcts-20k-s7, 210, 0, 2
- bench-v3-20k-s7, 210, 0, 5
- bench-mcts-20k-p32-s0, 210, 0, 2
- bench-mcts-20k-p32-s1, 210, 0, 2
- bench-mcts-20k-p32-s2, 210, 0, 2
- bench-mcts-20k-p32-s3, 210, 0, 2
- bench-r2t4-20k-s0, 1676, 124, 29
- bench-r2t5-20k-s0, 1676, 124, 28
- bench-r2t4-20k-s1, 1676, 124, 29
- bench-r2t5-20k-s1, 1676, 124, 28
- bench-mcts-1k-s0, 1676, 124, 2
- bench-mcts-1k-s1, 1676, 124, 2
- bench-mcts-2k-s0, 1676, 124, 2
- bench-mcts-2k-s1, 1676, 124, 2
- bench-mcts-10k-s0, 1676, 124, 7
- bench-mcts-10k-s1, 1676, 124, 8
- bench-mcts-50k-s0, 1676, 124, 30
- bench-mcts-50k-s1, 1676, 124, 30
- bench-mcts-5k-s0, 565, 35, 1
- bench-mcts-5k-s1, 565, 35, 1
- bench-v3-5k-s0, 565, 35, 3
- bench-v3-5k-s1, 565, 35, 3
- bench-v3-50k-s0, 565, 35, 21
- bench-v3-50k-s1, 565, 35, 21
- bench-r2t5-20k-s4, 210, 0, 4
- bench-r2t5-20k-s5, 210, 0, 5
- bench-r2t5-20k-s6, 210, 0, 5
- bench-r2t5-20k-s7, 210, 0, 4
- bench-confirm-mcts-20k-s0, 1163, 99, 9
- bench-confirm-r2t5-20k-s0, 1163, 99, 20
- bench-r2t5-20k-s2, 1676, 124, 28
- (training) bench-r2-t4, -, -, 10; bench-r2-t5, -, -, 15
- Not run: bench-r2t5-20k-s3 (replaced by Round 2c). No failures, no -r2 reruns, no gate skips.
