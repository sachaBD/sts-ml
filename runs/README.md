# runs/

Every job is a run: inputs in, outputs out. All results live here, one directory per run.

```
runs/
  run_id=<YYYY-MM-DD>_<kind>_<name>/
    run.json      written by the launcher (never by the job)
    logs/         stdout.log, stderr.log
    out/          everything the job produces
  scratch/run_id=.../   smoke / preflight / probe runs, same layout; safe to delete
```

- `kind`: `gen` (training data), `train` (checkpoint), `eval` (episodes / metrics).
- Runs are never modified after they finish. New results = new run.
- Scratch runs are outside the `runs/run_id=*` glob, so they never leak into queries.
- Parquet always sits under `out/schema=<name>/`, so a query names the schema it wants and never mixes layouts.

## Starting a run

```bash
PYTHONPATH=python .venv/bin/python -m sts_combat_rl.run --help
PYTHONPATH=python .venv/bin/python -m sts_combat_rl.run gen slime-pbcs20k-gen1 \
    --input 2026-09-22_train_gen0b-value -- build/generate_entry_mcts_records --out {out} ...
```

The launcher creates the directory, writes `run.json` (`status: running`), sends the job's stdout/stderr to
`logs/`, and on exit sets `status` to `done` / `failed` with `exit_code`. `--input` records lineage (run_ids this
job read from). `--scratch` puts the run under `runs/scratch/`.

## Contract for jobs (writers)

A job only has to:

1. **Write outputs into the out dir it is given**: `{out}` in the command line, or `$RUN_OUT` in the environment
   (`$RUN_DIR`, `$RUN_ID` are also set). Never write outside it. Don't create the run dir yourself.
2. **Log to stdout/stderr**: no log files, no pid files.
3. **Optionally write `out/summary.json`**: any JSON object (config such as teacher / simulations / seeds, plus
   results such as rows / episodes / wins). The launcher merges it into `run.json` as `summary`.

Git state, command, timings, host, and inputs are recorded by the launcher, so jobs don't need to.

### Output layout by kind

- **gen**: hive-partitioned parquet under a schema dir:
  `out/schema=<schema>/<key>=<value>/.../part-<NNN>.parquet`, e.g.
  `out/schema=combat_v1/act=1/floor=16/encounter=slime_boss/part-000.parquet`.
  - `schema` names the column layout. Adding a nullable column keeps the name (`union_by_name` fills NULLs);
    renaming, removing or changing the meaning of a column means a new name (`combat_v2`).
  - Partition keys after `schema` are fixed per schema and low-cardinality only (act, floor, encounter), never episode ids.
  - Don't also store `run_id` / `schema` / partition keys as columns; they come from the path.
  - Several parts per partition are fine (one per worker); aim for ~100 MB to 1 GB per part.
- **train**: `out/value_checkpoint.pt` (+ `.json` sidecar), metrics on stdout.
- **eval**: `out/episodes.jsonl` (one JSON object per fight), metrics in `summary.json`.

## run.json

| field | meaning |
|---|---|
| `run_id`, `kind`, `scratch`, `note` | identity |
| `status` | `running` / `done` / `failed` |
| `started`, `finished`, `exit_code`, `host`, `pid`, `cwd` | execution |
| `command` | argv of the job (with `{out}` substituted) |
| `inputs` | run_ids (or absolute paths) this run read from |
| `git` | `{repo: {rev, dirty}}` for sts_combat_rl, sts_lightspeed, sts_ml |
| `summary` | contents of `out/summary.json`, or null |

## Querying (duckdb)

```sql
-- all combat training data
-- (union_by_name: older runs lack newer columns, e.g. gen0 has no row_kind / parent_action; they read as NULL)
SELECT * FROM read_parquet('runs/run_id=*/out/schema=combat_v1/**/*.parquet', hive_partitioning = true, union_by_name = true);

-- which schemas exist, in which runs
-- (one read_parquet = one schema: each schema has its own partition keys and columns)
SELECT DISTINCT regexp_extract(file, 'run_id=([^/]+)', 1) AS run_id, regexp_extract(file, 'schema=([^/]+)', 1) AS schema
FROM glob('runs/run_id=*/out/schema=*/**/*.parquet');

-- all runs
SELECT run_id, kind, status, inputs, summary FROM read_json('runs/run_id=*/run.json');

-- all eval episodes, tagged with run
SELECT * FROM read_json('runs/run_id=*_eval_*/out/episodes.jsonl', filename = true, union_by_name = true);
```

## Schemas

| schema | status | written by | layout |
|---|---|---|---|
| `combat_v1` | current | `apps/generate_entry_mcts_records` | `act=/floor=/encounter=` partitions; columns below |
| `entry_roots_v1` | legacy | old `generate_entry_mcts_records` (entry-root bootstrap) | single part; `mcts_value`, `root_visits`, `replicate`, ... |
| `mcts_slime_v2` | legacy | `generate_mcts_records` (fixed Slime Boss deck, encoding v2) | single part |
| `mcts_slime_v1` | legacy | `generate_mcts_records` pilot | single part |

New schema = new row here.

### `combat_v1` columns

One row per decision recorded from teacher search.

| Column | Meaning |
|---|---|
| `episode_id` | fight within the run |
| `decision_index` | decision number within the fight, from 0 |
| `turn` | combat turn |
| `entry_id`, `deck_signature` | the starting deck/state |
| `combat_seed` | seed; with the start state and the chosen actions, replays the fight exactly |
| `starting_hp`, `starting_max_hp` | player HP at fight start |
| `encoding_version`, `global_numeric`, `cards`, `monsters`, `card_monster_interactions`, `input_state`, `card_selection_task` | encoded public state |
| `actions` | every move the search tried: `{action, description, visits, mean_value}` |
| `chosen_action` | move played |
| `was_random` | the one random move of the fight (not the teacher's choice) |
| `root_value` | search's value estimate for this state |
| `row_kind`, `parent_action` | `decision`: a position the teacher played from. `child`: the position after a move the teacher tried (≥ 50 visits) but didn't play; `parent_action` is that move, `root_value` is the teacher's mean value for it, and `actions`/`chosen_action` are empty |
| `simulations_used` | simulations the search actually ran: 500 for a forced move, fewer than the budget when it stopped early because the top move could no longer be overtaken; 0 on child rows |
| `won`, `final_hp`, `potions`, `terminal_value` | fight outcome, the same on every row of the fight (`potions`: potion count at fight end) |

`terminal_value`: win = (35 + final_hp + 4 × potions) / (55 + max_hp), loss = 0 (sts_ml `scorePrediction`, default weights).

Training targets are computed from these columns at training time. They aren't stored. Child rows are not on the
played trajectory, so their fight outcome columns don't apply to them; train them on `root_value`.
