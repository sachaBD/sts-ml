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
