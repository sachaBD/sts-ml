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
