# Stage 1: restart boundary screen — completed

Twelve deterministically selected training starts on which teacher correction won after
historical learner failure. Three nonterminal teacher-trajectory restarts per start,
four public-particle roots per restart, two continuation agents. All selected roots had
no unresolved known Headbutt top-order information under the history audit.

| Restart | Learner2k wins | Teacher20k wins |
|---|---:|---:|
| Early | 28/48 | 35/48 |
| Middle | 30/48 | 37/48 |
| Late | 41/48 | 48/48 |

288/288 main continuations completed; zero errors/caps. The48 observations per row are
12 trajectory clusters x4 particles, not48 independent fights. This is a selected training
screen, not a general performance estimate. Per-position paired teacher advantage is
7/48 (14.6 percentage points) in each row, but its variability differs across trajectories.
Only four particles per state: no individual-state mastery claim.

All seven late learner losses occur on two trajectories: learner-7:85 (4/4 losses) and
learner-11:2 (3/4). The learner can already finish most other late positions. Therefore
Stage2 starts with75% middle and25% late practice, rather than exclusively late practice.
This is a curriculum hypothesis, not evidence that training on these positions will help
complete fights.

## Operational record

- Pilot manifest sha prefix10ec8549;36 unique restart states, none terminal/known-order.
- Gate fixes: RNG fingerprint changed to explicit fields; harness missing-binary/fail-fast
  defects corrected; accidentally introduced tree reuse removed to match original fresh
  tree per decision. No such gameplay change is part of the experiment.
- Reproduction budget explicitly amended12->15:5 frozen-worker,5 failed app,5 corrected
  app attempts. Final learner action/visit/value checks matched. Teacher smoke8.
- Main run log: launched approximately14:06 BST, ended14:08:32 BST on2026-10-06,
  exit0. These are observed timestamps, not future runtime bounds.
- Completion communication arrived substantially later. Cause is not established by the
  logs; future chained runner must generate its final report automatically.

## Artifacts

Root: `runs/schema=combat_v4/date=2026-10-06/id=counterfactual-rescue-v1/`

- `out/starts.json`, `out/restart-history-audit.json`
- `out/gate-attempt-002/summary.json`, `out/frozen/`
- `out/stage1-results.jsonl` (raw), `out/stage1-config.json`
- `out/stage1-summary.json`, `out/STAGE1.md` (aggregate/per-trajectory report)
- `logs/stage1.log`

No training occurred in Stage1. No reserved final seeds were used.
