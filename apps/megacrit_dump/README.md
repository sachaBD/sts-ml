# megacrit_dump

Pulls a small, filtered slice of the public Mega Crit Slay the Spire 1 run-history dump (Google Drive, ~380 GB) into
Parquet. Raw files are downloaded into memory only and never stored. Schema: `runs/schema=megacrit_runs_v1/`.

```sh
# 1. list (one small run; recurses into subfolders; flags folders hitting Drive's 5,500-entry listing cap)
PYTHONPATH=. .venv/bin/python -m runs.run megacrit_runs_v1 listing --no-compact -- \
    .venv/bin/python apps/megacrit_dump/list.py --out {out}
# 2. pull N files spread evenly over the date range (resumable, <=4 workers; exit 3 if Drive throttles)
PYTHONPATH=. .venv/bin/python -m runs.run megacrit_runs_v1 <id> --no-compact --input <listing run id> -- \
    .venv/bin/python apps/megacrit_dump/pull.py --files <listing out>/files.parquet --out {out} \
    --n-files 50 --filter-character IRONCLAD --filter-ascension 20
```

Kept: the character/ascension filters and not daily/endless/trial/chose_seed; deduplicated by `play_id`.
`out/runs/part-*.parquet` columns: play_id, source_file_id, source_name, timestamp, build_version, character, ascension,
victory, floor_reached, seed_played, reached_champ, champ_won, champ_floor, champ_damage, champ_turns, hp_before_champ,
max_hp_before_champ (`*_per_floor[champ_floor-2]`, the state entering the fight), raw (full record, JSON).
`out/done/` records processed files (resume); `out/summary.json` has totals. Tests: `test_megacrit_dump.py`.
