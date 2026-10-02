# Act 2 expert iteration — controller report

Status: running
Elapsed: 2.26 h
Incumbent: `/home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=act2-r00-train/out/model.pt`

Score = 0.4 floor/33 + 0.2 [act 1 boss beaten] + 0.4 [act 2 boss beaten]. ± = 1 SE of paired diffs.

| Comparison | n | Cand score | Inc score | Δscore ± SE | Δact1 ± SE | Δact2 ± SE | Promoted |
|---|---:|---:|---:|---:|---:|---:|---|
| round 0 | 600 | 0.545 | 0.546 | -0.001 ± 0.010 | -7.8 ± 1.6 pp | +2.8 ± 1.7 pp | True |

| Play | n | score | act1 clear | act2 clear | mean floor |
|---|---:|---:|---:|---:|---:|
| act2-r00-collect | 3000 | 0.480 | 83.7% | 5.5% | 24.0 |
| act2-dev-joint-1001-overworld-r00 | 600 | 0.546 | 93.7% | 10.7% | 26.1 |
| act2-dev-act2-r00-train | 600 | 0.545 | 85.8% | 13.5% | 26.3 |
