# Act 2 expert iteration — controller report

Status: running
Elapsed: 6.59 h
Incumbent: `/home/sborowsk/project/sts_combat_rl/runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt`

Score = 0.4 floor/33 + 0.2 [act 1 boss beaten] + 0.4 [act 2 boss beaten]. ± = 1 SE of paired diffs.

| Comparison | n | Cand score | Inc score | Δscore ± SE | Δact1 ± SE | Δact2 ± SE | Promoted |
|---|---:|---:|---:|---:|---:|---:|---|
| round 0 | 600 | 0.545 | 0.546 | -0.001 ± 0.010 | -7.8 ± 1.6 pp | +2.8 ± 1.7 pp | True |
| round 1 | 600 | 0.567 | 0.545 | +0.022 ± 0.011 | -1.0 ± 1.6 pp | +2.7 ± 1.8 pp | True |
| round 2 | 600 | 0.593 | 0.567 | +0.026 ± 0.011 | +0.3 ± 1.5 pp | +5.7 ± 1.7 pp | True |

| Play | n | score | act1 clear | act2 clear | mean floor |
|---|---:|---:|---:|---:|---:|
| act2-r00-collect | 3000 | 0.480 | 83.7% | 5.5% | 24.0 |
| act2-dev-joint-1001-overworld-r00 | 600 | 0.546 | 93.7% | 10.7% | 26.1 |
| act2-dev-act2-r00-train | 600 | 0.545 | 85.8% | 13.5% | 26.3 |
| act2-r01-collect | 3000 | 0.477 | 77.3% | 8.4% | 23.9 |
| act2-dev-act2-r01-train | 600 | 0.567 | 84.8% | 16.2% | 27.4 |
| act2-r02-collect | 3000 | 0.494 | 73.7% | 12.2% | 24.5 |
| act2-dev-act2-r02-train | 600 | 0.593 | 85.2% | 21.8% | 27.6 |
