"""megacrit_runs_v1: Slay the Spire 1 player runs sliced from the public Mega Crit run-history dump.

Producer: apps/megacrit_dump (list.py -> `files`; pull.py -> `runs`, `done`). The raw dump is never stored.
Layout (not `<table>-*.parquet`; launcher run with --no-compact so names stay stable):
  out/files.parquet        listing run: the crawled Drive files
  out/runs/part-*.parquet  pull run: one row per kept run (deduplicated by play_id)
  out/done/part-*.parquet  pull run: per-file processing record, used for resume and summary totals
"""
import pyarrow as pa

NAME = "megacrit_runs_v1"

FILES = pa.schema([
    ("file_id", pa.string()),            # Drive file id
    ("name", pa.string()),               # e.g. 2018-10-25-02-34#1352.json.gz
    ("folder_path", pa.string()),        # "" for the root folder, else e.g. "Monthly_2020_10"
    ("timestamp", pa.timestamp("s")),    # parsed from the name (UTC, minute resolution); null if unparsable
])

RUNS = pa.schema([
    ("play_id", pa.string()),
    ("source_file_id", pa.string()),
    ("source_name", pa.string()),
    ("timestamp", pa.int64()),           # run record `timestamp` (epoch seconds) as stored; null if absent
    ("build_version", pa.string()),
    ("character", pa.string()),          # character_chosen
    ("ascension", pa.int32()),           # ascension_level
    ("victory", pa.bool_()),
    ("floor_reached", pa.int32()),
    ("seed_played", pa.string()),
    ("reached_champ", pa.bool_()),       # a damage_taken entry with enemies == "Champ" (the dump's name; "The Champ" also accepted)
    ("champ_won", pa.bool_()),           # reached_champ and (victory or floor_reached > champ_floor)
    ("champ_floor", pa.int32()),         # null unless reached_champ
    ("champ_damage", pa.int32()),
    ("champ_turns", pa.int32()),
    ("hp_before_champ", pa.int32()),     # current_hp_per_floor[champ_floor - 2]: HP after floor F-1, i.e. entering F
    ("max_hp_before_champ", pa.int32()), # max_hp_per_floor[champ_floor - 2]
    ("raw", pa.string()),                # the full run record (the `event` object) as JSON
])

DONE = pa.schema([
    ("file_id", pa.string()),
    ("runs_seen", pa.int64()),
    ("runs_kept", pa.int64()),           # after filters and play_id dedupe
    ("kept_reached_champ", pa.int64()),
    ("kept_champ_won", pa.int64()),
    ("bytes", pa.int64()),               # compressed bytes downloaded
])
