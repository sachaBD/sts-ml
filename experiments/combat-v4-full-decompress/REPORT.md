# combat_v4_full decompressor

`build/decompress/combat_v4_decompress --out DIR --fights F.parquet [--search S.parquet] [--part K --parts N]`
Input: combat_v4 fights (+ search) rows, selected beforehand with DuckDB (`scratch`-style `copy (select ... where start.encounter = 39)`; see `select_fights.py`). Output: `decisions-*.parquet`, `fights-*.parquet`
(schema.py exactly), `mismatches.txt`, `summary.json`. All C++ (Arrow/Parquet from the venv's pyarrow wheel), no JSON;
one process per core via `--part K --parts N`. Move descriptions were dropped from the schema.

Runs (ah-c01..c04, ah-dev-*, ah-fresh-*; 159 source files):
- `runs/schema=combat_v4_full/date=2026-10-03/id=ah-all`: 177,344 fights, 2,829,430 decisions, 0 replay mismatches,
  18 s wall with 3 processes (376 MB). id=ah-champ: 1,955 Champ fights, 83,282 decisions: DuckDB select 0.4 s + tool 0.6 s (19 MB).
- Previous Python/JSON version: 159 s for the 1.8k-fight Champ subset. The new one matches its output row for row
  (fights, state, legal, search) except `kind`, which is now lower-case as the contract says.

Champ: phase2 flips 0-1 turns after Champ first stands below 50% HP (max HP ratio at first sighting 0.498);
`champ_phase.py RUN_OUT` lists it per fight. Test: `apps/combat_record/test_decompress.py` (CTest).
Contract notes: search.agent lacks the `run_rl ` prefix of fights.agent (joined by suffix); lice keep a spent
Curl Up amount in uniquePower0 (decoded `curl_up_raw`); no other undecoded raw ints in this data (a crash flags any new one).
