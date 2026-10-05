# megacrit_runs_v1

Slay the Spire 1 human runs from the public Mega Crit run-history dump (Google Drive), filtered at pull time
(character, ascension, no daily/endless/trial/chosen-seed) so the ~380 GB dump is never stored. Contract:
[schema.py](schema.py). Tool: [`apps/megacrit_dump`](../../apps/megacrit_dump/README.md).

Tables: `files` (listing) -> `runs` (one row per kept run, `raw` = full record) and `done` (per-file bookkeeping).
`runs.source_file_id` joins `files.file_id`; `done.file_id` joins `files.file_id`. A pull run references its listing
run via `--input`. Keys: `files.file_id`; `runs.play_id` (unique within one pull run; different pull runs may overlap).
Drive caps folder listings at 5,500 entries, so the listing may be incomplete (see its summary.json).
