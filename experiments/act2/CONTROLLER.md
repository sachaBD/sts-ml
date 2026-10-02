# Act 2 expert iteration — controller report

Status: done
Elapsed: 2.26 h
Incumbent: `runs/schema=value_net_v1/date=2026-10-02/id=act2-r02-train/out/model.pt`

Score = 0.4 floor/33 + 0.2 [act 1 boss beaten] + 0.4 [act 2 boss beaten]. ± = 1 SE of paired diffs.

| Comparison | n | Cand score | Inc score | Δscore ± SE | Δact1 ± SE | Δact2 ± SE | Promoted |
|---|---:|---:|---:|---:|---:|---:|---|
| round 3 | 600 | 0.595 | 0.593 | +0.002 ± 0.011 | +0.5 ± 1.4 pp | +0.2 ± 1.8 pp | False |

| Play | n | score | act1 clear | act2 clear | mean floor |
|---|---:|---:|---:|---:|---:|
| act2b-r03-collect | 3000 | 0.556 | 79.8% | 19.3% | 26.3 |
| act2-dev-act2-r02-train | 600 | 0.593 | 85.2% | 21.8% | 27.6 |
| act2-dev-act2b-r03-train | 600 | 0.595 | 85.7% | 22.0% | 27.7 |
