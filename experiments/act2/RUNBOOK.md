
2026-10-02T07:40:57Z START act2-a1-rollout-baseline; 800 seeds from 950000000000, 10 workers; selected overworld r00; frozen joint controller worker; guided rollout (no combat leaf/weights); sims 500,5000,10000,10000,20000. Expected ~25 min.

2026-10-02T07:57:23Z FINISH act2-a1-rollout-baseline exit=0; launcher log experiments/act2/act2-a1-rollout-baseline.launcher.log

2026-10-02T07:57:41+00:00 RESULT {"n": 800, "candidate_clear": 0.91875, "incumbent_clear": 0.88625, "clear_diff": 0.0325, "clear_se": 0.010698025387144115, "score_diff": 0.03581500000000001, "score_se": 0.011097775788305684, "score_definition": "Act-1 clear + 0.1 * final HP / starting max HP; gate only, not training objective", "uncertainty": "SE over paired run seeds; approximate 95% intervals = mean +/- 1.96 SE", "comparison": "act2-a1-rollout-baseline minus joint-1001-fresh-final"}

2026-10-02T08:06:30Z SRE LAUNCH act2-controller pid=1240767 via nohup experiments/act2/launch.sh; expected ~9 h; launcher log experiments/act2/controller.launcher.log
2026-10-02T08:06:31+00:00 START act2-r00-collect
