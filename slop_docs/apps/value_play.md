# App spec: `apps/value_play/`: replay held-out fights with a teacher leaf strategy

## Purpose

Play the fights a value net was **not** trained on, with the teacher search that generated the
training data, with one change: search leaves are scored by the value net instead of by guided
rollouts (or, per config, by a bounded guided rollout then the value net, or by a fresh guided-rollout
baseline with no net). The output is a `combat_v3` run, so it can be compared with the stored teacher run
directly (see `compare_fights.md`) without re-running the teacher.

```
combat_v3 teacher run ──trained──▶ value_net_v1 run ──input──▶ value_play ──▶ combat_v3 run (NN teacher)
        └──────────────────── same episode_ids, compared by compare_fights ─────────────┘
```

## Usage

```bash
./apps/value_play/run.sh apps/value_play/slime.toml [--scratch] [--overwrite]
```

```toml
[run]
id = "slime-value-4-play"
input = "value_net_v1/2026-09-23/slime-value-4"
workers = 10
# optional (defaults = the original value_play):
leaf = "value_net"        # "value_net" (immediate), "hybrid", or "guided_rollout" (no net)
rollout_turns = 1         # hybrid only, >= 1: guided-rollout turn increments before the net
rollout_steps = 16        # hybrid only, >= 1: max guided-rollout actions before the net
random_move = true        # false: no random move (every move is the search's)
oracle = false            # true: search the TRUE state (1 particle, real RNG / draw order): perfect-foresight
                          # upper bound, not fair play. Log banner; rows tagged oracle = true (runs/README.md)
episodes = [707, 9305]    # subset of the checkpoint's validation episodes (default: all)
```

Invalid combinations fail instead of being ignored: unknown `[run]` keys, rollout bounds on
`value_net`/`guided_rollout`, `hybrid` without both bounds, episodes outside the validation set.
Pilot configs (all 14 validation fights, `random_move = false`, one worker): `slime4-R.toml`
(guided rollout), `slime4-N.toml` (value net), `slime4-H.toml` (hybrid 1 turn / 16 steps).

## What comes from where

| thing | source |
|---|---|
| value net weights | value run `run.json` summary `weights` (native C++ `ValueNet`) |
| fights | `validation_episode_ids` in the value run's `checkpoint_json`. `episode_id = run_seed * 100 + fight_index` |
| data runs | the value run's `run.json` `inputs`: one or more `combat_v3` runs |
| ascension, earlier fights | the data runs' `decision` rows (`ascension`, `chosen_action` by `decision_index`) |
| search settings | `agents/teacher_leaves.hpp`, recorded in `summary.json` `teacher` (incl. `random_move`) |
| random move | if `random_move` (default): as bootstrap, one per fight, from `mt19937_64(episode_id ^ 0xe9510)` |

Leaf strategies (`agents/teacher_leaves.hpp`; same particles, 15k simulations, forced moves 500):

- `guided_rollout`: the bootstrap teacher exactly (early stop every 500). No weights.
- `value_net`: leaves scored by `ValueNet` where expanded, batches of 64, early stop between batches.
- `hybrid`: as `value_net`, but each leaf is first played on by the guided rollout for at most
  `rollout_turns` turn increments / `rollout_steps` actions (`PublicBeliefCombatSearch::requestBatch`);
  a rollout that ends the fight is backed up with its terminal value, not scored by the net.

## Same fight

The worker rebuilds the act 1 run up to the fight: SimpleAgent out of combat (as bootstrap), the
earlier fights by their stored chosen actions (each fight must end exactly after its last stored
action). Then the value-net teacher plays the fight, and the worker stops.

Check per fight: the replayed `decision_index = 0` row must equal the stored one on `encounter`,
`floor`, `starting_hp`, `starting_max_hp`, `global_numeric`, `cards`, `monsters`. A mismatch fails
the run.

## Layout

```
apps/value_play/
  run.sh        CONFIG.toml [--scratch] → sts_combat_rl.run combat_v3 <id> --input <value run> --input <each data run>
  job.sh        build value_play_worker in build-valexp/ (never build/ or build-dev/), exec play.py
  play.py       resolve runs and fights; copy the worker into out/ (read-only; sha256 in summary);
                worker pool; parquet parts; start-state check; log; summary.json
  worker.cpp    value_play_worker WEIGHTS|--no-weights FIGHT.json OUTPUT_DIR → OUTPUT_DIR/fight.msgpack
  slime.toml, slime4-{R,N,H}.toml
```

Worker: `WEIGHTS` is required for `value_net`/`hybrid`; `--no-weights` for `guided_rollout`.
`FIGHT.json` = `{run_seed, ascension, fight_index, actions, teacher?}` with optional
`teacher = {leaf, rollout_turns, rollout_steps, random_move, oracle}`; without it: `value_net`, random move
on (the original behavior, so `WEIGHTS FIGHT.json OUTPUT_DIR` callers are unchanged).

Shared with bootstrap: `agents/teacher_leaves.{hpp,cpp}` (search, budget, leaf strategies),
`agents/teacher_search.{hpp,cpp}` (recording),
`scenarios/act1_run.{hpp,cpp}` (act 1 loop, fight columns), `apps/worker_io.hpp`,
`sts_combat_rl.schemas.combat_v3`, `sts_combat_rl.run.run_dir`.

## Output: `combat_v3` run

- `out/part-<episode_id>.parquet`: same columns and meaning as bootstrap, including child rows and
  the random move (if on). `simulations_used` counts the replay's simulations.
- `out/value_play_worker`: the worker binary that played the run.
- `out/summary.json`: `schema`, `value_run`, `data_runs`, `teacher` (settings), `episodes`,
  `worker` (`path`, `sha256`), `fights`, `wins`,
  `teacher_wins`, `rows`, `mean_terminal_value`, `teacher_mean_terminal_value`, `seconds_per_fight`.
- Log: one line per fight, the replay next to the stored teacher's result.

## Not in scope (v1)

- Statistics (`compare_fights`).
- Budget sweeps; mixed leaf mode.
- Fights other than the value run's validation set.
