# Runbook

2026-10-01 22:22 UTC: approved plan; implementation in progress. Existing run-RL v4 collected 2,000 runs but crashed on a stale import (current refactored trainer import resolves). No completed v4 checkpoint. Reuse v3 iter003 as initial overworld policy. Existing all-fight neural-leaf checkpoint ab-gen1; old full-run test at lower budgets was weaker than rollout combat (−4.3 points ±1.5 SE, n=1,000); new mixed budgets must be evaluated. User requests combat_v4 and new overworld_v1 recording, exploration and simple system.

2026-10-01 22:48 UTC: First 16-run exploratory pilot finished in 60s (4 workers), 196.8 total worker-seconds, mean 12.3 worker-s/run. n=16 too small for a reliable performance claim (10/16 clears). 129 fights, all 129 exact endpoint replay checks passed; 2,106 decision rows, 76 random combat actions; ~2.63 MB compressed combat tables. Previous chat count 83 was an unverified misstatement, corrected to 129 after querying Parquet.

Implementation: ordinary initial boundary only; std::function action queues cannot be serialized. Unsupported initialization boundaries retain outcome/provenance only and are explicitly excluded from replay facts and training cache (replay_error). Every accepted fight is replayed to exact final scalar/token/RNG snapshot equality. Derived cache uses legacy encoding shape without writing combat_v3 data. New combat stable validation partition prevents validation run seeds moving into training when replay window changes. Multi-table output compaction disabled tonight because legacy compactor does not preserve prefixes.

2026-10-01 22:54 UTC: Managed 64-run collection smoke passed (48/64 clear with both explorations, 14.0 worker-s/run mean, 180s on 6 workers). Combat_v4-to-trainer one-epoch fine-tune/export passed. Run-value training smoke underway. Added a deterministic public-state safety valve for documented neural defensive stalls: from turn 30, neural fights use 5k guided-rollout search. Actual results stay actual; no timeout converted to a loss. This is a mixed emergency fallback, recorded separately in fight results and disclosed in report, not a new architecture/budget sweep. Full run-level decisions and combat actions remain unchanged on ordinary shorter fights.

2026-10-01T22:54:02+00:00 START joint-1001-initial /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 910000000000 --seeds 400 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-initial --combat-leaf value_net --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin

2026-10-01 22:54 UTC: Overnight controller launched successfully through runs.run as overworld_v1/2026-10-01/joint-1001-controller. Hard deadline 2026-10-02 ~07:54 UTC (9 wall hours). First stage: 400 fresh development seeds with the frozen initial neural/overworld pair, followed by matched rollout-combat baseline at equal category budgets. Then sequential 1,500-run collection, combat fine-tune + paired gate, collection, overworld TD fine-tune + paired gate. Last training-independent confirmation reserved; optional last collection saves useful data for next iteration if another update cannot fit. Controller report and state initialized, all 10 CPU workers active. Logs: runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/logs/stdout.log; report experiments/joint-expert-iteration/REPORT.md. Training smoke succeeded for both model families; smoke weights are diagnostic only, not promoted.

2026-10-01T23:07:12+00:00 DONE joint-1001-initial

2026-10-01T23:07:12+00:00 START joint-1001-rollout /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 910000000000 --seeds 400 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-rollout

2026-10-01T23:15:04+00:00 DONE joint-1001-rollout

2026-10-01T23:15:06+00:00 START joint-1001-r00-a /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/collect.py --out {out} --id joint-1001-r00-a --first-seed 920000000000 --seeds 1500 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin --ckpt runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt

2026-10-02T00:02:55+00:00 DONE joint-1001-r00-a

2026-10-02T00:02:55+00:00 START joint-1001-combat-r00 /home/sborowsk/project/sts_combat_rl/.venv/bin/python -m apps.run_rl.train_combat --out {out} --init runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_checkpoint.pt --data combat_v4/2026-10-01/joint-1001-r00-a

2026-10-02T00:03:28+00:00 DONE joint-1001-combat-r00

2026-10-02T00:03:28+00:00 START joint-1001-r00-combat-base /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 930000000000 --seeds 400 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-r00-combat-base --combat-leaf value_net --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin

2026-10-02T00:15:31+00:00 DONE joint-1001-r00-combat-base

2026-10-02T00:15:31+00:00 START joint-1001-r00-combat-test /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 930000000000 --seeds 400 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-r00-combat-test --combat-leaf value_net --combat-weights /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-combat-r00/out/value_weights.bin

2026-10-02T00:27:32+00:00 DONE joint-1001-r00-combat-test

2026-10-02T00:27:33+00:00 GATE {"n": 400, "candidate_clear": 0.8775, "incumbent_clear": 0.8675, "clear_diff": 0.01, "clear_se": 0.013701069237311883, "score_diff": 0.010393333333333334, "score_se": 0.01408543893440802, "score_definition": "Act-1 clear + 0.1 * final HP / starting max HP; gate only, not training objective", "uncertainty": "SE over paired run seeds; approximate 95% intervals = mean +/- 1.96 SE", "name": "combat round 0", "promoted": false}

2026-10-02T00:27:33+00:00 START joint-1001-r00-b /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/collect.py --out {out} --id joint-1001-r00-b --first-seed 920000050000 --seeds 1500 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin --ckpt runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt

2026-10-02T01:15:06+00:00 DONE joint-1001-r00-b

2026-10-02T01:15:06+00:00 START joint-1001-overworld-r00 /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/train.py --out {out}/model.pt --init runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt --data /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-r00-a/out /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-02/id=joint-1001-r00-b/out --decay .85 --lam .7 --epochs 20

2026-10-02T01:15:56+00:00 DONE joint-1001-overworld-r00

2026-10-02T01:15:56+00:00 START joint-1001-r00-overworld-test /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 930000000000 --seeds 400 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-r00-overworld-test --combat-leaf value_net --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin

2026-10-02T01:28:28+00:00 DONE joint-1001-r00-overworld-test

2026-10-02T01:28:30+00:00 GATE {"n": 400, "candidate_clear": 0.9225, "incumbent_clear": 0.8675, "clear_diff": 0.055, "clear_se": 0.01886337900984797, "score_diff": 0.05983333333333333, "score_se": 0.019708572305816872, "score_definition": "Act-1 clear + 0.1 * final HP / starting max HP; gate only, not training objective", "uncertainty": "SE over paired run seeds; approximate 95% intervals = mean +/- 1.96 SE", "name": "overworld round 0", "promoted": true}

2026-10-02T01:28:30+00:00 START joint-1001-r01-a /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/collect.py --out {out} --id joint-1001-r01-a --first-seed 920000100000 --seeds 1500 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt

2026-10-02T02:15:46+00:00 DONE joint-1001-r01-a

2026-10-02T02:15:46+00:00 START joint-1001-combat-r01 /home/sborowsk/project/sts_combat_rl/.venv/bin/python -m apps.run_rl.train_combat --out {out} --init runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_checkpoint.pt --data combat_v4/2026-10-01/joint-1001-r00-a combat_v4/2026-10-02/joint-1001-r00-b combat_v4/2026-10-02/joint-1001-r01-a

2026-10-02T02:17:24+00:00 DONE joint-1001-combat-r01

2026-10-02T02:17:24+00:00 START joint-1001-r01-combat-base /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 930000100000 --seeds 400 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-r01-combat-base --combat-leaf value_net --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin

2026-10-02T02:30:01+00:00 DONE joint-1001-r01-combat-base

2026-10-02T02:30:01+00:00 START joint-1001-r01-combat-test /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 930000100000 --seeds 400 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-r01-combat-test --combat-leaf value_net --combat-weights /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-combat-r01/out/value_weights.bin

2026-10-02T02:42:09+00:00 DONE joint-1001-r01-combat-test

2026-10-02T02:42:10+00:00 GATE {"n": 400, "candidate_clear": 0.9275, "incumbent_clear": 0.9225, "clear_diff": 0.005, "clear_se": 0.01119154263568772, "score_diff": 0.005726666666666658, "score_se": 0.011472428265347168, "score_definition": "Act-1 clear + 0.1 * final HP / starting max HP; gate only, not training objective", "uncertainty": "SE over paired run seeds; approximate 95% intervals = mean +/- 1.96 SE", "name": "combat round 1", "promoted": false}

2026-10-02T02:42:10+00:00 START joint-1001-r01-b /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/collect.py --out {out} --id joint-1001-r01-b --first-seed 920000150000 --seeds 1500 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt

2026-10-02T03:29:13+00:00 DONE joint-1001-r01-b

2026-10-02T03:29:13+00:00 START joint-1001-overworld-r01 /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/train.py --out {out}/model.pt --init /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt --data /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-r00-a/out /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-02/id=joint-1001-r00-b/out /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-02/id=joint-1001-r01-a/out /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-02/id=joint-1001-r01-b/out --decay .85 --lam .7 --epochs 20

2026-10-02T03:30:58+00:00 DONE joint-1001-overworld-r01

2026-10-02T03:30:58+00:00 START joint-1001-r01-overworld-test /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 930000100000 --seeds 400 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r01/out/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-r01-overworld-test --combat-leaf value_net --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin

2026-10-02T03:43:01+00:00 DONE joint-1001-r01-overworld-test

2026-10-02T03:43:02+00:00 GATE {"n": 400, "candidate_clear": 0.9025, "incumbent_clear": 0.9225, "clear_diff": -0.02, "clear_se": 0.01657367541591122, "score_diff": -0.020136666666666674, "score_se": 0.01720298549905677, "score_definition": "Act-1 clear + 0.1 * final HP / starting max HP; gate only, not training objective", "uncertainty": "SE over paired run seeds; approximate 95% intervals = mean +/- 1.96 SE", "name": "overworld round 1", "promoted": false}

2026-10-02T03:43:02+00:00 START joint-1001-r02-a /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/collect.py --out {out} --id joint-1001-r02-a --first-seed 920000200000 --seeds 1500 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt

2026-10-02T04:29:39+00:00 DONE joint-1001-r02-a

2026-10-02T04:29:39+00:00 START joint-1001-combat-r02 /home/sborowsk/project/sts_combat_rl/.venv/bin/python -m apps.run_rl.train_combat --out {out} --init runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_checkpoint.pt --data combat_v4/2026-10-02/joint-1001-r01-a combat_v4/2026-10-02/joint-1001-r01-b combat_v4/2026-10-02/joint-1001-r02-a

2026-10-02T04:31:17+00:00 DONE joint-1001-combat-r02

2026-10-02T04:31:17+00:00 START joint-1001-r02-combat-base /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 930000200000 --seeds 400 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-r02-combat-base --combat-leaf value_net --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin

2026-10-02T04:43:55+00:00 DONE joint-1001-r02-combat-base

2026-10-02T04:43:55+00:00 START joint-1001-r02-combat-test /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 930000200000 --seeds 400 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-r02-combat-test --combat-leaf value_net --combat-weights /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-combat-r02/out/value_weights.bin

2026-10-02T04:56:33+00:00 DONE joint-1001-r02-combat-test

2026-10-02T04:56:34+00:00 GATE {"n": 400, "candidate_clear": 0.91, "incumbent_clear": 0.9, "clear_diff": 0.01, "clear_se": 0.013235859243918093, "score_diff": 0.010046666666666667, "score_se": 0.013665379041569368, "score_definition": "Act-1 clear + 0.1 * final HP / starting max HP; gate only, not training objective", "uncertainty": "SE over paired run seeds; approximate 95% intervals = mean +/- 1.96 SE", "name": "combat round 2", "promoted": false}

2026-10-02T04:56:34+00:00 START joint-1001-r02-b /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/collect.py --out {out} --id joint-1001-r02-b --first-seed 920000250000 --seeds 1500 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt

2026-10-02T05:44:34+00:00 DONE joint-1001-r02-b

2026-10-02T05:44:34+00:00 START joint-1001-overworld-r02 /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/train.py --out {out}/model.pt --init /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt --data /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-02/id=joint-1001-r01-a/out /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-02/id=joint-1001-r01-b/out /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-02/id=joint-1001-r02-a/out /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-02/id=joint-1001-r02-b/out --decay .85 --lam .7 --epochs 20

2026-10-02T05:46:10+00:00 DONE joint-1001-overworld-r02

2026-10-02T05:46:10+00:00 START joint-1001-r02-overworld-test /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 930000200000 --seeds 400 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r02/out/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-r02-overworld-test --combat-leaf value_net --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin

2026-10-02T05:58:37+00:00 DONE joint-1001-r02-overworld-test

2026-10-02T05:58:39+00:00 GATE {"n": 400, "candidate_clear": 0.9075, "incumbent_clear": 0.9, "clear_diff": 0.0075, "clear_se": 0.017871986312014736, "score_diff": 0.008056666666666665, "score_se": 0.018722624170613197, "score_definition": "Act-1 clear + 0.1 * final HP / starting max HP; gate only, not training objective", "uncertainty": "SE over paired run seeds; approximate 95% intervals = mean +/- 1.96 SE", "name": "overworld round 2", "promoted": false}

2026-10-02T05:58:39+00:00 START joint-1001-final-data /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/collect.py --out {out} --id joint-1001-final-data --first-seed 940000000000 --seeds 1500 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt

2026-10-02T06:46:00+00:00 DONE joint-1001-final-data

2026-10-02T06:46:00+00:00 Fresh final confirmation n=800

2026-10-02T06:46:00+00:00 START joint-1001-fresh-final /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 950000000000 --seeds 800 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt /home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=joint-1001-overworld-r00/out/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-fresh-final --combat-leaf value_net --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin

2026-10-02T07:11:58+00:00 DONE joint-1001-fresh-final

2026-10-02T07:11:58+00:00 START joint-1001-fresh-start /home/sborowsk/project/sts_combat_rl/.venv/bin/python apps/run_rl/play.py --out {out} --first-seed 950000000000 --seeds 800 --workers 10 --worker /home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker --policy net --ckpt runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt --decide rest path shop event neow --sims 500,5000,10000,10000,20000 --overworld-record --collection-id joint-1001-fresh-start --combat-leaf value_net --combat-weights runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin

2026-10-02T07:22:54+00:00 FAILED CalledProcessError(241, ['/home/sborowsk/project/sts_combat_rl/.venv/bin/python', '-m', 'runs.run', 'overworld_v1', 'joint-1001-fresh-start', '--no-compact', '--input', 'runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt', '--input', 'runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_checkpoint.pt', '--', '/home/sborowsk/project/sts_combat_rl/.venv/bin/python', 'apps/run_rl/play.py', '--out', '{out}', '--first-seed', '950000000000', '--seeds', '800', '--workers', '10', '--worker', '/home/sborowsk/project/sts_combat_rl/runs/schema=overworld_v1/date=2026-10-01/id=joint-1001-controller/out/run_rl_worker', '--policy', 'net', '--ckpt', 'runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt', '--decide', 'rest', 'path', 'shop', 'event', 'neow', '--sims', '500,5000,10000,10000,20000', '--overworld-record', '--collection-id', 'joint-1001-fresh-start', '--combat-leaf', 'value_net', '--combat-weights', 'runs/schema=value_net_v1/date=2026-09-27/id=ab-gen1/out/value_weights.bin'])

2026-10-02T07:23:50+00:00 STOPPED BY USER: active baseline and workers terminated by explicit PID; controller exited on child interruption. No remaining experiment processes. Partial outputs preserved; no fresh paired conclusion. Concise REPORT.md finalized. No new jobs scheduled.
