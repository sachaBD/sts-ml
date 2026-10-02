
2026-10-02T07:40:57Z START act2-a1-rollout-baseline; 800 seeds from 950000000000, 10 workers; selected overworld r00; frozen joint controller worker; guided rollout (no combat leaf/weights); sims 500,5000,10000,10000,20000. Expected ~25 min.

2026-10-02T07:57:23Z FINISH act2-a1-rollout-baseline exit=0; launcher log experiments/act2/act2-a1-rollout-baseline.launcher.log

2026-10-02T07:57:41+00:00 RESULT {"n": 800, "candidate_clear": 0.91875, "incumbent_clear": 0.88625, "clear_diff": 0.0325, "clear_se": 0.010698025387144115, "score_diff": 0.03581500000000001, "score_se": 0.011097775788305684, "score_definition": "Act-1 clear + 0.1 * final HP / starting max HP; gate only, not training objective", "uncertainty": "SE over paired run seeds; approximate 95% intervals = mean +/- 1.96 SE", "comparison": "act2-a1-rollout-baseline minus joint-1001-fresh-final"}

2026-10-02T08:06:30Z SRE LAUNCH act2-controller pid=1240767 via nohup experiments/act2/launch.sh; expected ~9 h; launcher log experiments/act2/controller.launcher.log
2026-10-02T08:06:31+00:00 START act2-r00-collect
2026-10-02T09:36:33+00:00 DONE act2-r00-collect
2026-10-02T09:36:43+00:00 START act2-r00-train
2026-10-02T09:37:57+00:00 DONE act2-r00-train
2026-10-02T09:37:57+00:00 START act2-dev-joint-1001-overworld-r00
2026-10-02T09:57:59+00:00 DONE act2-dev-joint-1001-overworld-r00
2026-10-02T09:58:00+00:00 START act2-dev-act2-r00-train
2026-10-02T10:22:04+00:00 DONE act2-dev-act2-r00-train
2026-10-02T10:22:11+00:00 GATE {"candidate": {"n": 600, "score": 0.5447777777777778, "act1": 0.8583333333333333, "act2": 0.135, "floor": 26.326666666666668}, "incumbent": {"n": 600, "score": 0.545979797979798, "act1": 0.9366666666666666, "act2": 0.10666666666666667, "floor": 26.06833333333333}, "score_diff": -0.0012020202020201981, "score_se": 0.010444480282594447, "act1_diff": -0.07833333333333334, "act1_se": 0.015764538769630406, "act2_diff": 0.028333333333333332, "act2_se": 0.01672374755089808, "floor_diff": 0.25833333333333336, "floor_se": 0.32649194531304165, "name": "round 0", "promoted": true}
2026-10-02T10:22:11+00:00 START act2-r01-collect
2026-10-02T12:04:55+00:00 DONE act2-r01-collect
2026-10-02T12:05:04+00:00 START act2-r01-train
2026-10-02T12:09:10+00:00 DONE act2-r01-train
2026-10-02T12:09:12+00:00 START act2-dev-act2-r01-train
2026-10-02T12:32:36+00:00 DONE act2-dev-act2-r01-train
2026-10-02T12:32:44+00:00 GATE {"candidate": {"n": 600, "score": 0.5668989898989899, "act1": 0.8483333333333334, "act2": 0.16166666666666665, "floor": 27.436666666666667}, "incumbent": {"n": 600, "score": 0.5447777777777778, "act1": 0.8583333333333333, "act2": 0.135, "floor": 26.326666666666668}, "score_diff": 0.022121212121212125, "score_se": 0.010604913851766996, "act1_diff": -0.01, "act1_se": 0.015819305211110583, "act2_diff": 0.02666666666666667, "act2_se": 0.01777661840033325, "floor_diff": 1.11, "floor_se": 0.3096425677376214, "name": "round 1", "promoted": true}
2026-10-02T12:32:44+00:00 START act2-r02-collect
2026-10-02T14:12:31+00:00 DONE act2-r02-collect
2026-10-02T14:12:40+00:00 START act2-r02-train
2026-10-02T14:18:26+00:00 DONE act2-r02-train
2026-10-02T14:18:29+00:00 START act2-dev-act2-r02-train
2026-10-02T14:41:55+00:00 DONE act2-dev-act2-r02-train
2026-10-02T14:42:01+00:00 GATE {"candidate": {"n": 600, "score": 0.5927979797979799, "act1": 0.8516666666666667, "act2": 0.21833333333333332, "floor": 27.648333333333333}, "incumbent": {"n": 600, "score": 0.5668989898989899, "act1": 0.8483333333333334, "act2": 0.16166666666666665, "floor": 27.436666666666667}, "score_diff": 0.025898989898989894, "score_se": 0.010633039440392521, "act1_diff": 0.0033333333333333335, "act1_se": 0.015468327391507303, "act2_diff": 0.056666666666666664, "act2_se": 0.017179640515113493, "floor_diff": 0.21166666666666667, "floor_se": 0.3086497346352495, "name": "round 2", "promoted": true}
2026-10-02T14:42:01+00:00 START act2-fresh-final
2026-10-02T15:14:08+00:00 DONE act2-fresh-final
2026-10-02T15:14:10+00:00 START act2-fresh-initial
2026-10-02T15:40:35+00:00 DONE act2-fresh-initial
2026-10-02T15:40:44+00:00 FRESH {"candidate": {"n": 800, "score": 0.5869166666666668, "act1": 0.83875, "act2": 0.21, "floor": 27.65125}, "incumbent": {"n": 800, "score": 0.5385075757575758, "act1": 0.91625, "act2": 0.1075, "floor": 25.76125}, "score_diff": 0.048409090909090915, "score_se": 0.00993390455795844, "act1_diff": -0.0775, "act1_se": 0.014650443059932194, "act2_diff": 0.1025, "act2_se": 0.016855186416543145, "floor_diff": 1.89, "floor_se": 0.27961152203381673, "name": "FRESH selected vs initial"}

2026-10-02T15:40:45Z SRE FINISH act2-controller exit=0

2026-10-02T15:41:12Z SRE LAUNCH act2b-controller pid=1245718 via nohup experiments/act2/launch2.sh; expected ~3.3 h; launcher log experiments/act2/controller2.launcher.log
2026-10-02T15:41:13+00:00 START act2b-r03-collect
2026-10-02T17:30:01+00:00 DONE act2b-r03-collect
2026-10-02T17:30:11+00:00 START act2b-r03-train
2026-10-02T17:35:01+00:00 DONE act2b-r03-train
2026-10-02T17:35:03+00:00 START act2-dev-act2b-r03-train
2026-10-02T17:56:58+00:00 DONE act2-dev-act2b-r03-train
2026-10-02T17:57:04+00:00 GATE {"candidate": {"n": 600, "score": 0.5952121212121213, "act1": 0.8566666666666667, "act2": 0.22, "floor": 27.71}, "incumbent": {"n": 600, "score": 0.5927979797979799, "act1": 0.8516666666666667, "act2": 0.21833333333333332, "floor": 27.648333333333333}, "score_diff": 0.002414141414141415, "score_se": 0.010648169981092031, "act1_diff": 0.005, "act1_se": 0.014444355264196827, "act2_diff": 0.0016666666666666668, "act2_se": 0.01834850386648401, "floor_diff": 0.06166666666666667, "floor_se": 0.2759952816153531, "name": "round 3", "promoted": false}

2026-10-02T17:57:04Z SRE FINISH act2b-controller exit=0
