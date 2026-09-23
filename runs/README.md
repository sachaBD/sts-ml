# runs/

Every job is a run: inputs in, outputs out. All results live here, one directory per run.

```
runs/schema=<schema>/date=<YYYY-MM-DD>/id=<id>/
  run.json         written by the launcher (never by the job)
  logs/            stdout.log, stderr.log
  out/             everything the job produces; optional summary.json
scratch/schema=<schema>/date=<YYYY-MM-DD>/id=<id>/   same layout; smoke / preflight / probe runs; safe to delete
```

| level | form | rule |
|---|---|---|
| root | `runs/` or `scratch/` | `scratch/` is never queried |
| schema | `schema=<schema>` | what the run produces. Exactly one per run, declared at launch. A schema with no table is valid (no data) |
| date | `date=<YYYY-MM-DD>` | UTC start date, set by the launcher |
| id | `id=<id>` | free text, `a-z0-9_.-`, describes the run. Unique within schema + date |
| run.json | `run.json` | launcher only |
| logs | `logs/` | the job's stdout and stderr |
| out | `out/` | the only place the job writes |

- **run_id** is `<schema>/<date>/<id>`, e.g. `combat_v2/2026-09-22/slime-gen0` → `runs/schema=combat_v2/date=2026-09-22/id=slime-gen0/`.
  It is what `--input`, `inputs` in `run.json` and notes refer to.
- Runs are never modified after they finish. New results = new run.
- A job that produces two schemas is two runs, the second taking the first as `--input`.
- `schema`, `date` and `id` come from the path; don't also store them as columns.

## Starting a run

```bash
PYTHONPATH=python .venv/bin/python -m sts_combat_rl.run --help
./apps/bootstrap/run.sh apps/bootstrap/act1.toml [--scratch]
```

The launcher creates the directory, writes `run.json` (`status: running`), sends the job's stdout/stderr to
`logs/`, and on exit sets `status` to `done` / `failed` with `exit_code`. `--input` records lineage (run_ids this
job read from). `--scratch` puts the run under `scratch/`.

## Contract for jobs (writers)

A job only has to:

1. **Write outputs into the out dir it is given**: `{out}` in the command line, or `$RUN_OUT` in the environment
   (`$RUN_DIR`, `$RUN_ID` are also set). Never write outside it. Don't create the run dir yourself.
2. **Log to stdout/stderr**: no log files, no pid files.
3. **Optionally write `out/summary.json`**: any JSON object (config such as teacher / simulations / seeds, plus
   results such as rows / episodes / wins). The launcher merges it into `run.json` as `summary`.

Git state, command, timings, host, and inputs are recorded by the launcher, so jobs don't need to.

### Tables

- Parquet goes directly in `out/` as `part-<NNN>.parquet` (`combat_v3`: one per seeded run, `NNN` = run seed).
- No partition directories. Encounter, seed, etc. are ordinary columns; sort or group rows by the column you
  filter on most so duckdb can skip row groups.
- Adding a nullable column keeps the schema name (`union_by_name` fills NULLs). Renaming, removing or changing
  the meaning of a column means a new name (`combat_v3`).

## run.json

| field | meaning |
|---|---|
| `run_id`, `schema`, `scratch`, `note` | identity |
| `status` | `running` / `done` / `failed` |
| `started`, `finished`, `exit_code`, `host`, `pid`, `cwd` | execution |
| `command` | argv of the job (with `{out}` substituted) |
| `inputs` | run_ids (or absolute paths) this run read from |
| `git` | `{repo: {rev, dirty}}` for sts_combat_rl, sts_lightspeed, sts_ml |
| `summary` | contents of `out/summary.json`, or null |

## Querying (duckdb)

```sql
-- all combat training data, with schema / date / id columns from the path
-- (union_by_name: older runs lack newer columns; they read as NULL)
SELECT * FROM read_parquet('runs/schema=combat_v3/*/*/out/*.parquet', hive_partitioning = true, union_by_name = true);

-- rows per category / encounter
SELECT category, encounter, count(*) FROM read_parquet('runs/schema=combat_v3/*/*/out/*.parquet') GROUP BY ALL ORDER BY ALL;

-- all runs
SELECT run_id, schema, status, inputs, summary FROM read_json('runs/*/*/*/run.json');

-- all eval episodes, tagged with run
SELECT * FROM read_json('runs/schema=episodes_v1/*/*/out/episodes.jsonl', filename = true, union_by_name = true);
```

## Schemas

| schema | status | written by | out/ |
|---|---|---|---|
| `combat_v3` | current | `apps/bootstrap/generate.py` | parquet; columns below |
| `combat_v2` | legacy | `apps/bootstrap/generate.py` before combat_v3 | Slime Boss fights only; `combat_seed` instead of `run_seed`, `episode_id` = seed, legacy `entry_id`/`deck_signature`, no `category`/`fight_index`/`ascension` |
| `combat_v1` | legacy | `apps/bootstrap/generate.py` before combat_v2 | as `combat_v2`, but `act`/`floor`/`encounter`/`seed` were partition directories |
| `value_net_v1` | current | `sts_combat_rl.training.train_value` | `value_checkpoint.pt` + `.json` sidecar (+ `value_weights.bin`) |
| `episodes_v1` | current | eval jobs | `episodes.jsonl`, one JSON object per fight; metrics in `summary.json` |
| `entry_roots_v1` | legacy | old entry-root bootstrap writer | single part; `mcts_value`, `root_visits`, `replicate`, ... |
| `mcts_slime_v2` | legacy | `generate_mcts_records` (fixed Slime Boss deck, encoding v2) | single part |
| `mcts_slime_v1` | legacy | `generate_mcts_records` pilot | single part |

New schema = new row here.

### `combat_v3` columns

Name and pyarrow layout: `apps/bootstrap/schema.py` (`NAME`, `COMBAT_V3`); each part also stores `schema=combat_v3`
in its parquet metadata. One seeded Ironclad act 1 per run seed: seeds whose act 1 boss isn't Slime Boss are skipped
at game creation (no file); SimpleAgent plays everything out of combat; the teacher search plays **every combat**
until the player dies or beats Slime Boss. One row per decision recorded from teacher search, in play order
(`fight_index`, then decision rows, then child rows).

| Column | Meaning |
|---|---|
| `run_seed` | game seed; with the chosen actions, replays the whole run exactly. Groups the fights of one run |
| `episode_id` | one fight: `run_seed * 100 + fight_index` |
| `fight_index` | fight number within the run, from 0 |
| `act`, `floor` | where the fight is |
| `encounter` | lightspeed encounter, lowercase: `cultist`, `gremlin_nob`, `lagavulin`, `slime_boss`, ... |
| `category` | `easy` (act's weak hallway pool: the first 3 hallway fights), `hard` (strong hallway pool), `elite`, `boss`, `event` (fight started from a `?` room event) |
| `ascension` | ascension level of the run |
| `starting_hp`, `starting_max_hp` | player HP at fight start |
| `decision_index` | decision number within the fight, from 0 |
| `turn` | combat turn |
| `row_kind`, `parent_action` | `decision`: a position the teacher played from. `child`: the position after a move the teacher tried (≥ 50 visits) but didn't play; `parent_action` is that move, `root_value` is the teacher's mean value for it, and `actions`/`chosen_action` are empty |
| `encoding_version`, `global_numeric`, `cards`, `monsters`, `card_monster_interactions`, `input_state`, `card_selection_task` | encoded public state |
| `actions` | every move the search tried: `{action, description, visits, mean_value}` |
| `chosen_action` | move played |
| `was_random` | the one random move of the fight (not the teacher's choice); a fight shorter than 24 decisions may have none |
| `root_value` | search's value estimate for this state |
| `simulations_used` | simulations the search actually ran: 500 for a forced move, fewer than the budget when it stopped early because the top move could no longer be overtaken; 0 on child rows |
| `won`, `final_hp`, `potions`, `terminal_value` | this fight's outcome, the same on every row of the fight (`potions`: potion count at fight end) |

`terminal_value`: win = (35 + final_hp + 4 × potions) / (55 + max_hp), loss = 0 (sts_ml `scorePrediction`, default weights).

Training targets are computed from these columns at training time. They aren't stored. `load_rows` can keep only
some `categories` / `encounters`; `episode_split` splits by `run_seed`, so a run's fights stay on one side. Child rows are not on the
played trajectory, so their fight outcome columns don't apply to them; train them on `root_value`.
