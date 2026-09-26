# slime-v8 expert iteration runbook (for orchestrator-26)

Approved by the owner 2026-09-26. Execute generations in order, one job at a time, 10 workers each, from the
repository root. Background: `REPORT.md` (generation 1 beat the rollout teacher on the dev fights).

## Terms

- **Current player:** the network that generates self-play data and that each candidate must beat.
  At the start: **generation 1**, value run `value_net_v1/2026-09-25/slime-v8-gen1`, dev run
  `combat_v3/2026-09-25/slime-v8-gen1-dev`.
- **Rollout teacher dev run:** `combat_v3/2026-09-25/slime-v8-rollout-teacher-dev` (fixed; never re-run).
- **Dev fights:** the 999 fights in `SPLIT.md`. **Confirm fights: never play or train on them.**

## One generation N (~70 min)

| step | command | approx time |
|---|---|---:|
| 1 self-play | `./apps/fight_resample/run.sh rundecks/slime-v8/configs/slime-v8-selfplay-genN.toml` | 30 min |
| 2 fine-tune | `./apps/value_train/run.sh rundecks/slime-v8/configs/slime-v8-genN.toml` | 3 min |
| 3 dev fights | `./apps/value_play/run.sh rundecks/slime-v8/configs/slime-v8-genN-dev.toml` | 35 min |
| 4 compare | see below | 1 min |

Step 4:
```bash
.venv/bin/python rundecks/slime-v8/compare.py <current player dev run> <genN dev run> --json rundecks/slime-v8/genN-vs-current.json
.venv/bin/python rundecks/slime-v8/compare.py combat_v3/2026-09-25/slime-v8-rollout-teacher-dev <genN dev run> --json rundecks/slime-v8/genN-vs-rollout-teacher.json
```

**Promotion rule (mechanical):** if the `difference` in `genN-vs-current.json` is > 0, generation N becomes the
current player (its value run and dev run). Otherwise the current player is unchanged. Either way, continue.

Append to `REPORT.md` a short section per generation: self-play wins / fights and mean terminal value; the two
comparison lines (difference, 95% CI, wins only each side); promoted or not; run ids. Numbers only, no analysis.

## Configs for generation N >= 3

Generation 2's configs are written (`configs/slime-v8-*gen2*.toml`). For N >= 3, copy generation N-1's three
configs and change only:

- every `genN-1` in ids / comments -> `genN`;
- self-play `query` decks and `first_sample` from the schedule below;
- self-play `value_run` and fine-tune `initial_checkpoint` -> the **current player's** value run id;
- fine-tune query: add `'slime-v8-selfplay-genN'` to the `id in (...)` list (all self-play runs so far);
- dev config `input` -> `value_net_v1/<date>/slime-v8-genN` with the date the fine-tune run actually got.

Run ids carry the UTC start date (`runs/schema=<schema>/date=<date>/id=<id>`): check the real path of each run
before referencing it.

## Deck schedule

| gen | self-play decks (`encounter = 'slime_boss' and source_episode_id is null and ...`) | first_sample |
|---|---|---:|
| 1 (done) | `id = 'act1-a20-8' and run_seed % 10 in (4, 5)` | 0 |
| 2 | `id = 'act1-a20-8' and run_seed % 10 in (6, 7)` | 0 |
| 3 | `id = 'act1-a20-8' and run_seed % 10 in (8, 9)` | 0 |
| 4 | `id in ('act1-a20', 'act1-a20-1', 'act1-a20-2', 'act1-a20-5', 'act1-a20-6', 'act1-a20-7')` | 0 |
| 5 | `id = 'act1-a20-8' and run_seed % 10 in (4, 5)` | 1 |
| 6 | `id = 'act1-a20-8' and run_seed % 10 in (6, 7)` | 1 |
| 7 | `id = 'act1-a20-8' and run_seed % 10 in (8, 9)` | 1 |

Never use `run_seed % 10 in (0, 1, 2, 3)` of act1-a20-8 (dev / confirm).

## Stopping

Keep going through the schedule until the owner stops it or compute runs out. No stopping on results.
Don't start a generation you are told will not have time to finish.

## Failures

Fix scripting / environment / small errors yourself and rerun the failed step (`--overwrite` for the same run id).
Wake the coordinator only for blockers: anything that would change the method (settings, data, rules above),
a repeated failure of the same step, or results that look broken (e.g. a run with far fewer fights than expected).

## Notify

Do not message the coordinator per generation: REPORT.md is the record. Message only for blockers (above).
