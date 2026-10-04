2026-10-03T23:28:40Z START play champ-bench-teacher20k starts=runs/schema=combat_v4/date=2026-10-04/id=champ-bench/out/fights-00000.parquet worker=build/frozen/pv_worker.t1 agent=teacher sims=20000 workers=8 
2026-10-03T23:28:40Z START train champ-boot-w64 data=runs/schema=pv_rows_v1/date=2026-10-04/id=champ-train-teacher/out/rows.parquet args=--epochs 25 --width 64
2026-10-03T23:35:58Z DONE train champ-boot-w64
2026-10-03T23:40:01Z START train champ-boot-w64-t025 data=runs/schema=pv_rows_v1/date=2026-10-04/id=champ-train-teacher/out/rows.parquet args=--epochs 6 --width 64 --policy-temp 0.25
2026-10-03T23:41:51Z DONE train champ-boot-w64-t025
2026-10-03T23:41:51Z DONE play champ-bench-teacher20k
2026-10-03T23:42:01Z START play champ-bench-boot-w64-s2000 starts=runs/schema=combat_v4/date=2026-10-04/id=champ-bench/out/fights-00000.parquet worker=build/frozen/pv_worker agent=pv sims=2000 workers=9 --model runs/schema=pv_model_v1/date=2026-10-03/id=champ-boot-w64/out/model.onnx
2026-10-03T23:43:03Z START play champ-bench-boot-w64-s2000 starts=runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out/fights-00000.parquet worker=build/frozen/pv_worker agent=pv sims=2000 workers=9 --model runs/schema=pv_model_v1/date=2026-10-03/id=champ-boot-w64/out/model.onnx
2026-10-03T23:48:34Z DONE play champ-bench-boot-w64-s2000
2026-10-03T23:49:00Z START play champ-sp1-r01-play model=runs/schema=pv_model_v1/date=2026-10-03/id=champ-boot-w64/out/model.onnx sims=800 explore=True
2026-10-04T00:01:26Z DONE play champ-sp1-r01-play {  "fights": 3000,  "wins": 463,  "capped": 0,  "decision_seconds": 5796.395216533006,  "wall_seconds": 746.4283132553101,  "workers": 8,  "command": [   "play",   "runs/schema=pv_model_v1/date=2026-10-03/id=champ-boot-w64/out/model.onnx",   "800",   "--explore"  ],  "mean_seconds_per_fight": 1.9321317388443353 }
2026-10-04T00:01:34Z START train champ-sp1-r01 data=/home/sborowsk/project/sts_combat_rl/runs/schema=pv_rows_v1/id=champ-sp1-r01/rows.parquet args=--init runs/schema=pv_model_v1/date=2026-10-03/id=champ-boot-w64/out/model.pt --epochs 4
2026-10-04T00:02:58Z DONE train champ-sp1-r01
2026-10-04T00:02:58Z START play champ-sp1-r01-bench model=/home/sborowsk/project/sts_combat_rl/runs/schema=pv_model_v1/date=2026-10-04/id=champ-sp1-r01/out/model.onnx sims=2000 explore=False
2026-10-04T00:07:45Z DONE play champ-sp1-r01-bench {  "fights": 409,  "wins": 102,  "capped": 0,  "decision_seconds": 2484.1067080150024,  "wall_seconds": 286.56568217277527,  "workers": 9,  "command": [   "play",   "/home/sborowsk/project/sts_combat_rl/runs/schema=pv_model_v1/date=2026-10-04/id=champ-sp1-r01/out/model.onnx",   "2000"  ],  "mean_seconds_per_fight": 6.0736105330440155 }
2026-10-04T00:07:45Z BENCH champ-sp1-r01: /home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-sp1-r01-bench/compare.md
2026-10-04T00:07:49Z START play champ-sp1-r02-play model=/home/sborowsk/project/sts_combat_rl/runs/schema=pv_model_v1/date=2026-10-04/id=champ-sp1-r01/out/model.onnx sims=800 explore=True
2026-10-04T00:08:08Z START play champ-bench-boot-q1-c0.5 starts=runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out/fights-00000.parquet worker=build/frozen/pv_worker.q1 agent=pv sims=2000 workers=9 --model runs/schema=pv_model_v1/date=2026-10-03/id=champ-boot-w64/out/model.onnx --c 0.5
2026-10-04T00:14:22Z DONE play champ-bench-boot-q1-c0.5
2026-10-04T00:14:22Z START play champ-bench-boot-q1-c1.25 starts=runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out/fights-00000.parquet worker=build/frozen/pv_worker.q1 agent=pv sims=2000 workers=9 --model runs/schema=pv_model_v1/date=2026-10-03/id=champ-boot-w64/out/model.onnx --c 1.25
2026-10-04T00:20:44Z DONE play champ-bench-boot-q1-c1.25
2026-10-04T00:20:44Z START play champ-bench-boot-q1-c3 starts=runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out/fights-00000.parquet worker=build/frozen/pv_worker.q1 agent=pv sims=2000 workers=9 --model runs/schema=pv_model_v1/date=2026-10-03/id=champ-boot-w64/out/model.onnx --c 3
2026-10-04T00:27:11Z DONE play champ-bench-boot-q1-c3
2026-10-04T00:35:02Z START play champ-sp2-r01-play model=runs/schema=pv_model_v1/date=2026-10-04/id=champ-sp1-r01/out/model.onnx sims=800 explore=True
2026-10-04T00:50:26Z DONE play champ-sp2-r01-play {  "fights": 3000,  "wins": 527,  "capped": 0,  "decision_seconds": 7138.822693792992,  "wall_seconds": 922.8905210494995,  "workers": 8,  "command": [   "play",   "runs/schema=pv_model_v1/date=2026-10-04/id=champ-sp1-r01/out/model.onnx",   "800",   "--explore"  ],  "mean_seconds_per_fight": 2.379607564597664 }
2026-10-04T00:50:35Z START train champ-sp2-r01 data=/home/sborowsk/project/sts_combat_rl/runs/schema=pv_rows_v1/id=champ-sp2-r01/rows.parquet args=--init runs/schema=pv_model_v1/date=2026-10-04/id=champ-sp1-r01/out/model.pt --epochs 2
2026-10-04T00:51:36Z DONE train champ-sp2-r01
2026-10-04T00:51:36Z START play champ-sp2-r01-bench model=/home/sborowsk/project/sts_combat_rl/runs/schema=pv_model_v1/date=2026-10-04/id=champ-sp2-r01/out/model.onnx sims=2000 explore=False
2026-10-04T00:57:55Z DONE play champ-sp2-r01-bench {  "fights": 409,  "wins": 87,  "capped": 0,  "decision_seconds": 3227.1674933479976,  "wall_seconds": 379.3287823200226,  "workers": 9,  "command": [   "play",   "/home/sborowsk/project/sts_combat_rl/runs/schema=pv_model_v1/date=2026-10-04/id=champ-sp2-r01/out/model.onnx",   "2000"  ],  "mean_seconds_per_fight": 7.890385069310508 }
2026-10-04T00:57:55Z BENCH champ-sp2-r01: /home/sborowsk/project/sts_combat_rl/runs/schema=combat_v4/date=2026-10-04/id=champ-sp2-r01-bench/compare.md
2026-10-04T00:57:59Z START play champ-sp2-r02-play model=/home/sborowsk/project/sts_combat_rl/runs/schema=pv_model_v1/date=2026-10-04/id=champ-sp2-r01/out/model.onnx sims=800 explore=True
2026-10-04T00:59:04Z START train champ-boot-troot data=runs/schema=pv_rows_v1/date=2026-10-04/id=champ-train-teacher/out/rows.parquet args=--epochs 8 --width 64 --teacher-root-mix 0
2026-10-04T01:15:05Z START train champ-boot-troot data=runs/schema=pv_rows_v1/date=2026-10-04/id=champ-train-teacher/out/rows.parquet args=--epochs 8 --width 64 --teacher-root-mix 0
2026-10-04T01:15:41Z START play champ-teachergen-a starts=runs/schema=pv_starts_v1/id=champ-teachergen-a/starts.parquet worker=build/frozen/pv_worker.q1 agent=teacher sims=20000 workers=8 
2026-10-04T01:17:32Z DONE train champ-boot-troot
2026-10-04T01:17:46Z START play champ-bench-troot-c1.25 starts=runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out/fights-00000.parquet worker=build/frozen/pv_worker.q1 agent=pv sims=2000 workers=9 --model runs/schema=pv_model_v1/date=2026-10-04/id=champ-boot-troot/out/model.onnx
2026-10-04T01:22:47Z DONE play champ-bench-troot-c1.25
2026-10-04T01:22:47Z BENCH champ-bench-troot-c1.25 | all | 409 | 0.418 | 0.308 | -7.44 ± 1.27 | 64 / 19 | 0.000 |
2026-10-04T01:23:14Z START train champ-boot-troot-w128 data=runs/schema=pv_rows_v1/date=2026-10-04/id=champ-train-teacher/out/rows.parquet args=--epochs 8 --width 128 --teacher-root-mix 0
2026-10-04T01:23:14Z START train champ-boot-troot-m05 data=runs/schema=pv_rows_v1/date=2026-10-04/id=champ-train-teacher/out/rows.parquet args=--epochs 8 --width 64 --teacher-root-mix 0.5
2026-10-04T01:26:41Z DONE train champ-boot-troot-w128
2026-10-04T01:26:41Z DONE train champ-boot-troot-m05
2026-10-04T02:33:02Z DONE play champ-teachergen-a
