# Champ oracle expert iteration — runbook

Objective: `pv_champ_win_v3`, value target 100·won, softplus head and scaled MSE (not BCE).
Oracle tree reuse is path-local; no transpositions or hidden-state observation-key merging.
Turns histograms measure player END_TURN crossings per simulation; p90 is not p90 of decision maxima.

2026-10-04T09:59:05Z DONE iteration-0 encode champ-train contract=pv_champ_win_v3
2026-10-04T09:59:16Z DONE iteration-0 encode champ-teachergen-a contract=pv_champ_win_v3
2026-10-04T10:02:29Z DONE freeze build/frozen/pv_worker.ox-v3-precommit (rebuild+tests pass)
2026-10-04T10:03:09Z DONE tests native=pv_features_test,encoding_v4_test final-build=passed pipeline=40-fight-teacher-encode-train-real-oracle-policy-replay-passed logs=experiments/champ-oracle-exit/{final-build,tests}.log
2026-10-04T10:07:19Z DONE iteration-0 train width=64 epochs=10 fresh-init best-val-selection value=MSE target=100*won policy=teacher-visits log=runs/schema=combat_v4/date=2026-10-04/id=champ-ox-iteration-0/bootstrap.log
2026-10-04T10:08:06Z LAUNCH smoke (oracle800,real2000,replay) 9 workers, experiments/champ-oracle-exit/smoke.sh
2026-10-04T10:08:18Z DONE smoke oracle800 rc=0
2026-10-04T10:08:37Z DONE smoke real2000 rc=0
2026-10-04T10:08:38Z DONE smoke oracle800-replay rc=0
2026-10-04T10:08:39Z DONE smoke real2000-replay rc=0
2026-10-04T10:10:12Z LAUNCH sweep0 policy
2026-10-04T10:10:13Z DONE sweep0 policy rc=0
2026-10-04T10:10:13Z LAUNCH sweep0 oracle200
2026-10-04T10:10:51Z DONE sweep0 oracle200 rc=0
2026-10-04T10:10:51Z LAUNCH sweep0 oracle800
2026-10-04T10:13:26Z DONE sweep0 oracle800 rc=0
2026-10-04T10:13:26Z LAUNCH sweep0 real2000
2026-10-04T10:20:11Z DONE sweep0 real2000 rc=0
2026-10-04T10:20:11Z LAUNCH sweep0 oracle3200
2026-10-04T10:32:03Z DONE sweep0 oracle3200 rc=0
2026-10-04T10:32:58Z LAUNCH exit tag=a 3 rounds n=8000 sims=800 pid=51575 log=experiments/champ-oracle-exit/exit-a.log
2026-10-04T10:33:01Z DONE id=champ-ox-a-r01 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-a-r01/starts.log
2026-10-04T10:41:16Z STOPPED exit tag=a (round1 play ~1280/8000 fights done, 238 wins) by main request; partial outputs kept
2026-10-04T10:41:19Z LAUNCH exit tag=b 9 rounds n=3000 sims=800 real-every=3 pid=57057 log=experiments/champ-oracle-exit/exit-b.log
2026-10-04T10:41:22Z DONE id=champ-ox-b-r01 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-b-r01/starts.log
2026-10-04T10:48:45Z STOPPED exit tag=b (round1 play partial; outcome-only value, superseded by tag c value-mix 0.5) by main
2026-10-04T10:53:12Z DONE id=champ-ox-c-r01 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r01/starts.log
2026-10-04T10:53:36Z LAUNCH exit tag=c 9 rounds n=3000 sims=800 epochs=1 value-mix=0.5 real-every=3 commit=fc9bd18 worker=ox-e902f9a pid=62446 log=experiments/champ-oracle-exit/exit-c.log
2026-10-04T11:09:31Z DONE id=champ-ox-c-r01 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r01/play.log
2026-10-04T11:09:40Z DONE id=champ-ox-c-r01 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r01/encode.log
2026-10-04T11:09:47Z FAIL id=champ-ox-c-r01 train exit=1 log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r01/train.log
2026-10-04T11:16:52Z FAIL exit tag=c r01 train: ValueError value mixing needs PV search root values in [0,100]
2026-10-04T11:18:21Z LAUNCH exit tag=c RESUME r01 (play/encode skipped) 9 rounds n=3000 sims=800 epochs=1 value-mix=0.5 real-every=3 commit=7523def worker=ox-e902f9a pid=76284 log=experiments/champ-oracle-exit/exit-c.log
2026-10-04T11:19:00Z DONE id=champ-ox-c-r01 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r01/train.log
2026-10-04T11:22:17Z DONE id=champ-ox-c-r01 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r01/bench-oracle.log
2026-10-04T11:22:18Z DONE id=champ-ox-c-r01 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r01/bench-policy.log
2026-10-04T11:22:22Z DONE id=champ-ox-c-r02 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r02/starts.log
2026-10-04T11:44:51Z DONE id=champ-ox-c-r02 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r02/play.log
2026-10-04T11:45:01Z DONE id=champ-ox-c-r02 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r02/encode.log
2026-10-04T11:47:25Z DONE id=champ-ox-c-r02 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r02/train.log
2026-10-04T11:51:47Z DONE id=champ-ox-c-r02 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r02/bench-oracle.log
2026-10-04T11:51:48Z DONE id=champ-ox-c-r02 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r02/bench-policy.log
2026-10-04T11:51:52Z DONE id=champ-ox-c-r03 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r03/starts.log
2026-10-04T12:18:36Z DONE id=champ-ox-c-r03 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r03/play.log
2026-10-04T12:18:47Z DONE id=champ-ox-c-r03 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r03/encode.log
2026-10-04T12:22:28Z DONE id=champ-ox-c-r03 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r03/train.log
2026-10-04T12:26:50Z DONE id=champ-ox-c-r03 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r03/bench-oracle.log
2026-10-04T12:38:03Z DONE id=champ-ox-c-r03 bench-real log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r03/bench-real.log
2026-10-04T12:38:04Z DONE id=champ-ox-c-r03 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r03/bench-policy.log
2026-10-04T12:38:04Z DONE id=champ-ox-c-r03 compare log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r03/compare.log
2026-10-04T12:38:08Z DONE id=champ-ox-c-r04 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r04/starts.log
2026-10-04T12:59:41Z DONE id=champ-ox-c-r04 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r04/play.log
2026-10-04T12:59:53Z DONE id=champ-ox-c-r04 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r04/encode.log
2026-10-04T13:04:09Z DONE id=champ-ox-c-r04 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r04/train.log
2026-10-04T13:07:51Z DONE id=champ-ox-c-r04 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r04/bench-oracle.log
2026-10-04T13:07:52Z DONE id=champ-ox-c-r04 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r04/bench-policy.log
2026-10-04T13:07:58Z DONE id=champ-ox-c-r05 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r05/starts.log
2026-10-04T13:30:03Z DONE id=champ-ox-c-r05 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r05/play.log
2026-10-04T13:30:14Z DONE id=champ-ox-c-r05 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r05/encode.log
2026-10-04T13:34:26Z DONE id=champ-ox-c-r05 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r05/train.log
2026-10-04T13:39:48Z DONE id=champ-ox-c-r05 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r05/bench-oracle.log
2026-10-04T13:39:49Z DONE id=champ-ox-c-r05 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r05/bench-policy.log
2026-10-04T13:39:53Z DONE id=champ-ox-c-r06 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r06/starts.log
2026-10-04T14:04:16Z DONE id=champ-ox-c-r06 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r06/play.log
2026-10-04T14:04:27Z DONE id=champ-ox-c-r06 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r06/encode.log
2026-10-04T14:08:18Z DONE id=champ-ox-c-r06 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r06/train.log
2026-10-04T14:12:21Z DONE id=champ-ox-c-r06 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r06/bench-oracle.log
2026-10-04T14:24:23Z DONE id=champ-ox-c-r06 bench-real log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r06/bench-real.log
2026-10-04T14:24:24Z DONE id=champ-ox-c-r06 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r06/bench-policy.log
2026-10-04T14:24:24Z DONE id=champ-ox-c-r06 compare log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r06/compare.log
2026-10-04T14:24:28Z DONE id=champ-ox-c-r07 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r07/starts.log
2026-10-04T14:46:24Z DONE id=champ-ox-c-r07 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r07/play.log
2026-10-04T14:46:36Z DONE id=champ-ox-c-r07 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r07/encode.log
2026-10-04T14:50:45Z DONE id=champ-ox-c-r07 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r07/train.log
2026-10-04T14:54:18Z DONE id=champ-ox-c-r07 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r07/bench-oracle.log
2026-10-04T14:54:19Z DONE id=champ-ox-c-r07 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r07/bench-policy.log
2026-10-04T14:54:23Z DONE id=champ-ox-c-r08 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r08/starts.log
2026-10-04T15:16:01Z DONE id=champ-ox-c-r08 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r08/play.log
2026-10-04T15:16:13Z DONE id=champ-ox-c-r08 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r08/encode.log
2026-10-04T15:20:23Z DONE id=champ-ox-c-r08 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r08/train.log
2026-10-04T15:24:06Z DONE id=champ-ox-c-r08 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r08/bench-oracle.log
2026-10-04T15:24:08Z DONE id=champ-ox-c-r08 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r08/bench-policy.log
2026-10-04T15:24:11Z DONE id=champ-ox-c-r09 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r09/starts.log
2026-10-04T15:45:49Z DONE id=champ-ox-c-r09 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r09/play.log
2026-10-04T15:46:01Z DONE id=champ-ox-c-r09 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r09/encode.log
2026-10-04T15:48:23Z DONE id=champ-ox-c-r09 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r09/train.log
2026-10-04T15:51:42Z DONE id=champ-ox-c-r09 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r09/bench-oracle.log
2026-10-04T16:00:11Z DONE id=champ-ox-c-r09 bench-real log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r09/bench-real.log
2026-10-04T16:00:12Z DONE id=champ-ox-c-r09 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r09/bench-policy.log
2026-10-04T16:00:12Z DONE id=champ-ox-c-r09 compare log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r09/compare.log
2026-10-04T16:00:30Z DONE exit tag=c 9 rounds
2026-10-04T16:33:45Z LAUNCH exit tag=c r10-r18 n=6000 sims=800 workers=8 epochs=1 value-mix=0.5 real-every=3 commit=9c81956 worker=ox-e902f9a pid=228686 log=experiments/champ-oracle-exit/exit-c.log
2026-10-04T16:33:50Z DONE id=champ-ox-c-r10 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r10/starts.log
2026-10-04T17:17:25Z DONE id=champ-ox-c-r10 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r10/play.log
2026-10-04T17:17:49Z DONE id=champ-ox-c-r10 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r10/encode.log
2026-10-04T17:23:53Z DONE id=champ-ox-c-r10 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r10/train.log
2026-10-04T17:28:30Z DONE id=champ-ox-c-r10 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r10/bench-oracle.log
2026-10-04T17:28:30Z DONE id=champ-ox-c-r10 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r10/bench-policy.log
2026-10-04T17:28:34Z DONE id=champ-ox-c-r11 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r11/starts.log
2026-10-04T17:54:12Z FAIL id=champ-ox-c-r11 play exit=1 log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r11/play.log
2026-10-04T18:57:51Z LAUNCH exit tag=c RESUME r11-r18 worker CHANGED e902f9a->ox-426c5f5 (sts_lightspeed 648229b fixes stale-task discard recovery/replay divergence behind r11 'invalid action 4'; combat_rl 426c5f5); r11 partial play moved to play-crashed-e902f9a; pid=298373 log=experiments/champ-oracle-exit/exit-c.log
2026-10-04T19:43:45Z DONE id=champ-ox-c-r11 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r11/play.log
2026-10-04T19:44:09Z DONE id=champ-ox-c-r11 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r11/encode.log
2026-10-04T19:51:20Z DONE id=champ-ox-c-r11 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r11/train.log
2026-10-04T19:55:39Z DONE id=champ-ox-c-r11 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r11/bench-oracle.log
2026-10-04T19:55:40Z DONE id=champ-ox-c-r11 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r11/bench-policy.log
2026-10-04T19:55:43Z DONE id=champ-ox-c-r12 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r12/starts.log
2026-10-04T20:42:37Z DONE id=champ-ox-c-r12 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r12/play.log
2026-10-04T20:43:01Z DONE id=champ-ox-c-r12 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r12/encode.log
2026-10-04T20:49:33Z DONE id=champ-ox-c-r12 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r12/train.log
2026-10-04T20:52:42Z DONE id=champ-ox-c-r12 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r12/bench-oracle.log
2026-10-04T21:02:23Z DONE id=champ-ox-c-r12 bench-real log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r12/bench-real.log
2026-10-04T21:02:24Z DONE id=champ-ox-c-r12 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r12/bench-policy.log
2026-10-04T21:02:24Z DONE id=champ-ox-c-r12 compare log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r12/compare.log
2026-10-04T21:02:28Z DONE id=champ-ox-c-r13 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r13/starts.log
2026-10-04T21:02:53Z LAUNCH turnbench baseline (1 worker, cpu11)
2026-10-04T21:31:39Z DONE turnbench baseline rc=0
2026-10-04T21:31:39Z LAUNCH turnbench ts64 (1 worker, cpu11)
2026-10-04T21:48:46Z DONE id=champ-ox-c-r13 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r13/play.log
2026-10-04T21:49:08Z DONE id=champ-ox-c-r13 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r13/encode.log
2026-10-04T21:58:04Z DONE id=champ-ox-c-r13 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r13/train.log
2026-10-04T22:02:51Z DONE id=champ-ox-c-r13 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r13/bench-oracle.log
2026-10-04T22:02:52Z DONE id=champ-ox-c-r13 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r13/bench-policy.log
2026-10-04T22:02:56Z DONE id=champ-ox-c-r14 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r14/starts.log
2026-10-04T22:03:43Z DONE turnbench ts64 rc=0
2026-10-04T22:03:43Z LAUNCH turnbench ts16 (1 worker, cpu11)
2026-10-04T22:21:44Z DONE turnbench ts16 rc=0
2026-10-04T22:21:44Z LAUNCH turnbench ts256 (1 worker, cpu11)
2026-10-04T22:47:32Z DONE id=champ-ox-c-r14 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r14/play.log
2026-10-04T22:47:51Z DONE id=champ-ox-c-r14 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r14/encode.log
2026-10-04T22:51:46Z DONE id=champ-ox-c-r14 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r14/train.log
2026-10-04T22:54:51Z DONE id=champ-ox-c-r14 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r14/bench-oracle.log
2026-10-04T22:54:52Z DONE id=champ-ox-c-r14 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r14/bench-policy.log
2026-10-04T22:54:58Z DONE id=champ-ox-c-r15 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r15/starts.log
2026-10-04T23:32:26Z DONE id=champ-ox-c-r15 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r15/play.log
2026-10-04T23:32:47Z DONE id=champ-ox-c-r15 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r15/encode.log
2026-10-04T23:34:54Z DONE turnbench ts256 rc=0
2026-10-04T23:36:19Z DONE id=champ-ox-c-r15 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r15/train.log
2026-10-04T23:39:45Z DONE id=champ-ox-c-r15 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r15/bench-oracle.log
2026-10-04T23:48:31Z DONE id=champ-ox-c-r15 bench-real log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r15/bench-real.log
2026-10-04T23:48:32Z DONE id=champ-ox-c-r15 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r15/bench-policy.log
2026-10-04T23:48:32Z DONE id=champ-ox-c-r15 compare log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r15/compare.log
2026-10-04T23:48:35Z DONE id=champ-ox-c-r16 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r16/starts.log
2026-10-05T00:24:09Z DONE id=champ-ox-c-r16 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r16/play.log
2026-10-05T00:24:27Z DONE id=champ-ox-c-r16 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r16/encode.log
2026-10-05T00:27:55Z DONE id=champ-ox-c-r16 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r16/train.log
2026-10-05T00:30:51Z DONE id=champ-ox-c-r16 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r16/bench-oracle.log
2026-10-05T00:30:52Z DONE id=champ-ox-c-r16 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r16/bench-policy.log
2026-10-05T00:30:55Z DONE id=champ-ox-c-r17 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r17/starts.log
2026-10-05T01:05:18Z DONE id=champ-ox-c-r17 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r17/play.log
2026-10-05T01:05:36Z DONE id=champ-ox-c-r17 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r17/encode.log
2026-10-05T01:08:59Z DONE id=champ-ox-c-r17 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r17/train.log
2026-10-05T01:12:32Z DONE id=champ-ox-c-r17 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r17/bench-oracle.log
2026-10-05T01:12:33Z DONE id=champ-ox-c-r17 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r17/bench-policy.log
2026-10-05T01:12:37Z DONE id=champ-ox-c-r18 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r18/starts.log
2026-10-05T01:48:09Z DONE id=champ-ox-c-r18 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r18/play.log
2026-10-05T01:48:27Z DONE id=champ-ox-c-r18 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r18/encode.log
2026-10-05T01:51:51Z DONE id=champ-ox-c-r18 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r18/train.log
2026-10-05T01:54:49Z DONE id=champ-ox-c-r18 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r18/bench-oracle.log
2026-10-05T02:02:40Z DONE id=champ-ox-c-r18 bench-real log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r18/bench-real.log
2026-10-05T02:02:41Z DONE id=champ-ox-c-r18 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r18/bench-policy.log
2026-10-05T02:02:41Z DONE id=champ-ox-c-r18 compare log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r18/compare.log
2026-10-05T02:03:05Z DONE exit tag=c r10-r18 (r18 real2000 39.1%)
2026-10-05T02:03:13Z LAUNCH J1 PIMC k4 e16 M=r15 9 workers runs/schema=combat_v4/date=2026-10-05/id=champ-pimc-r15-k4-e16
2026-10-05T02:19:12Z DONE J1 rc=0
2026-10-05T02:19:40Z LAUNCH J2 exit tag=d init=c-r18 9 rounds n=3000 sims=64 workers=8 turn-search+turn-targets grad-clip=1.0 commit=885f55c worker=turn-targets-dc8f9579d2b6 pid=515274 log=experiments/champ-oracle-exit/exit-d.log
2026-10-05T02:19:43Z DONE id=champ-ox-d-r01 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r01/starts.log
2026-10-05T02:46:18Z DONE id=champ-ox-d-r01 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r01/play.log
2026-10-05T02:46:29Z DONE id=champ-ox-d-r01 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r01/encode.log
2026-10-05T02:47:18Z DONE id=champ-ox-d-r01 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r01/train.log
2026-10-05T02:51:10Z DONE id=champ-ox-d-r01 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r01/bench-oracle.log
2026-10-05T02:51:11Z DONE id=champ-ox-d-r01 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r01/bench-policy.log
2026-10-05T02:51:14Z DONE id=champ-ox-d-r02 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r02/starts.log
2026-10-05T03:17:13Z DONE id=champ-ox-d-r02 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r02/play.log
2026-10-05T03:17:23Z DONE id=champ-ox-d-r02 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r02/encode.log
2026-10-05T03:18:50Z DONE id=champ-ox-d-r02 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r02/train.log
2026-10-05T03:22:33Z DONE id=champ-ox-d-r02 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r02/bench-oracle.log
2026-10-05T03:22:34Z DONE id=champ-ox-d-r02 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r02/bench-policy.log
2026-10-05T03:22:37Z DONE id=champ-ox-d-r03 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r03/starts.log
2026-10-05T03:49:00Z DONE id=champ-ox-d-r03 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r03/play.log
2026-10-05T03:49:10Z DONE id=champ-ox-d-r03 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r03/encode.log
2026-10-05T03:51:17Z DONE id=champ-ox-d-r03 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r03/train.log
2026-10-05T03:54:50Z DONE id=champ-ox-d-r03 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r03/bench-oracle.log
2026-10-05T04:01:49Z DONE id=champ-ox-d-r03 bench-real log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r03/bench-real.log
2026-10-05T04:01:50Z DONE id=champ-ox-d-r03 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r03/bench-policy.log
2026-10-05T04:01:50Z DONE id=champ-ox-d-r03 compare log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r03/compare.log
2026-10-05T04:01:54Z DONE id=champ-ox-d-r04 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r04/starts.log
2026-10-05T04:28:24Z DONE id=champ-ox-d-r04 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r04/play.log
2026-10-05T04:28:33Z DONE id=champ-ox-d-r04 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r04/encode.log
2026-10-05T04:30:35Z DONE id=champ-ox-d-r04 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r04/train.log
2026-10-05T04:34:25Z DONE id=champ-ox-d-r04 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r04/bench-oracle.log
2026-10-05T04:34:26Z DONE id=champ-ox-d-r04 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r04/bench-policy.log
2026-10-05T04:34:30Z DONE id=champ-ox-d-r05 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r05/starts.log
2026-10-05T05:01:56Z DONE id=champ-ox-d-r05 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r05/play.log
2026-10-05T05:02:06Z DONE id=champ-ox-d-r05 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r05/encode.log
2026-10-05T05:04:09Z DONE id=champ-ox-d-r05 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r05/train.log
2026-10-05T05:07:54Z DONE id=champ-ox-d-r05 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r05/bench-oracle.log
2026-10-05T05:07:55Z DONE id=champ-ox-d-r05 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r05/bench-policy.log
2026-10-05T05:07:58Z DONE id=champ-ox-d-r06 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r06/starts.log
2026-10-05T05:34:14Z DONE id=champ-ox-d-r06 play log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r06/play.log
2026-10-05T05:34:23Z DONE id=champ-ox-d-r06 encode log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r06/encode.log
2026-10-05T05:36:32Z DONE id=champ-ox-d-r06 train log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r06/train.log
2026-10-05T05:40:20Z DONE id=champ-ox-d-r06 bench-oracle log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r06/bench-oracle.log
2026-10-05T05:47:38Z DONE id=champ-ox-d-r06 bench-real log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r06/bench-real.log
2026-10-05T05:47:39Z DONE id=champ-ox-d-r06 bench-policy log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r06/bench-policy.log
2026-10-05T05:47:39Z DONE id=champ-ox-d-r06 compare log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r06/compare.log
2026-10-05T05:47:42Z DONE id=champ-ox-d-r07 starts log=/home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-05/id=champ-ox-d-r07/starts.log
2026-10-05T05:48:10Z STOPPED exit tag=d after r06 (real2000 r03 33.5%, r06 34.7% vs M 39.1%; policy-only regression) by main request; outputs kept
2026-10-05T05:48:28Z LAUNCH diag D1 train 
2026-10-05T05:48:28Z LAUNCH diag c18-ts64
2026-10-05T05:52:22Z DONE diag c18ts64 rc=0
2026-10-05T05:53:30Z DONE diag D1-train rc=0
2026-10-05T05:53:30Z LAUNCH diag D2 train 
2026-10-05T05:53:37Z LAUNCH diag D1 bench-real2000
2026-10-05T05:57:07Z DONE diag D2-train rc=0
2026-10-05T05:57:07Z LAUNCH diag D3 train --policy-weight 0
2026-10-05T06:00:52Z DONE diag D3-train rc=0
2026-10-05T06:00:52Z LAUNCH diag D4 train --value-weight 0
2026-10-05T06:01:17Z DONE diag D1-real rc=0
2026-10-05T06:01:18Z DONE diag D1-pol rc=0
2026-10-05T06:01:18Z LAUNCH diag D2 bench-real2000
2026-10-05T06:04:20Z DONE diag D4-train rc=0
2026-10-05T06:04:20Z LAUNCH diag D5 train 
2026-10-05T06:09:47Z DONE diag D2-real rc=0
2026-10-05T06:09:48Z DONE diag D2-pol rc=0
2026-10-05T06:09:48Z LAUNCH diag D3 bench-real2000
2026-10-05T06:13:13Z DONE diag D5-train rc=0
2026-10-05T06:16:38Z DONE diag D3-real rc=0
2026-10-05T06:16:39Z DONE diag D3-pol rc=0
2026-10-05T06:16:39Z LAUNCH diag D4 bench-real2000
2026-10-05T06:23:57Z DONE diag D4-real rc=0
2026-10-05T06:23:58Z DONE diag D4-pol rc=0
2026-10-05T06:23:58Z LAUNCH diag D5 bench-real2000
2026-10-05T06:31:18Z DONE diag D5-real rc=0
2026-10-05T06:31:19Z DONE diag D5-pol rc=0
2026-10-05T06:33:06Z LAUNCH J3 teacher 9 workers
2026-10-05T07:27:05Z DONE J3 teacher rc=0
2026-10-05T07:27:05Z LAUNCH J3 c18 9 workers
