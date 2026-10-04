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
