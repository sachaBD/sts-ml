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
