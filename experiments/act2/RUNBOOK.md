
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
