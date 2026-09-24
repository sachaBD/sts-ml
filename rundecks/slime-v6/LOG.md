# slime-v6 execution log

- 07:54 UTC step 1 launched: bootstrap act1-a20-7 (1,250 seeds)
- 08:09 step 1 done (14.7 min): act1-a20-7 380 runs, 323 Slime fights; compacted. Bootstrap total 781 Slime fights.
- 08:10 step 3 launched: slime-v6-gen0
- 08:14 step 3 done (~2 min): gen0 625 train / 156 test fights, valid_mse 0.0086 (baseline 0.050)
- 08:15 step 4a teacher-play done (2.4 min): 121/156, tv 0.427
- 08:19 step 4b gen0-play done (~4 min): 115/156, tv 0.401; compare: -0.027 (-0.055, +0.002)
- 08:21 step 5 launched: slime-v6-dagger1 (625 fights)
- 08:43 step 5 dagger1 done (21.4 min): 625/625 completed, learner won 441, disagreement 5,863/17,549 decisions (33%)
- 08:45 gen1-w25 and gen1-w50 training launched in parallel
- 08:47 gen1-w25 / w50 trained (~3 min each): pinned split, 0 bootstrap rows dropped, 17,549 correction rows
- 09:00 gen1 evaluated: w25 114/156 tv 0.400; w50 112/156 tv 0.399. Both equal to gen0 (diff about 0.00). Chose w25 (tie; more wins).
- 09:23 dagger2 done (20.7 min): 625/625, learner (gen1-w25) won 438, disagreement 32.7%. gen2 train+play+compare launched
- 09:28 gen2 done: 34,422 correction rows; 110/156 tv 0.393; vs gen1 -0.007 (-0.023, +0.007), 108/156 fights identical. dagger3 chain launched
- 09:59 dagger3 done (20.6 min): 625/625, 441 wins, disagreement 31.9%. gen3: 51,278 correction rows; 106/156 tv 0.382; vs gen2 -0.011; vs teacher -0.045 (p 0.018); vs gen0 -0.019 (-0.042, +0.003)
- 10:01 REPORT.md final. Launched act1-a20-8 bootstrap (forever) for the user to stop.
- 19:37 oracle upper bound: slime-v6-oracle-play (guided-rollout teacher, oracle = true: 1 particle = true state) on the same 156 test fights, 2.1 min: 148/156, tv 0.528
- 19:41 compares vs oracle for teacher, gen0, gen1-w25, gen1-w50, gen2, gen3 (fight_comparison_v1/…/slime-v6-*-vs-oracle); REPORT.md oracle section added
