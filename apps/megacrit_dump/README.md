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

## Champ start states (`champ_starts.py`, schema `megacrit_champ_v1`)

For every kept run that reached Champ, rebuilds deck (with upgrades), relics, HP and max HP at the *start* of the Champ
fight by undoing, backward from the final record, everything on/after the Champ floor. Names are mapped to
sts_lightspeed enums by parsing `../sts_lightspeed/include/constants/*.h` (no hand-written tables). Undos that cannot be
done cleanly are listed in `issues` and set `exact = false`, never guessed. Potions are not reconstructed (not stored).
Event logs omit "+N", so a card removed or duplicated by an event has an unknown upgrade state (`event_card_upgrade_unknown`).

```sh
PYTHONPATH=. .venv/bin/python -m runs.run megacrit_champ_v1 <id> --no-compact --input <pull run id> -- \
    .venv/bin/python apps/megacrit_dump/champ_starts.py --runs <pull run dir> [<more run dirs>] --out {out}
PYTHONPATH=. .venv/bin/python experiments/megacrit-champ/analyze.py   # -> experiments/megacrit-champ/REPORT.md
```
Columns: play_id, source_run_id, build_version, timestamp, F, champ_won, champ_damage, champ_turns, hp, max_hp, deck_raw,
deck (card, upgrades), relics_raw, relics, unmapped, exact, issues. Tests: `test_champ_starts.py`.
