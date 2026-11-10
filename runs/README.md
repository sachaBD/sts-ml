# runs

`runs/` holds immutable job outputs. `scratch/` has the same layout for disposable work and is not queried.

```text
runs/schema=<schema>/
  schema.py                         table definitions and query registration
  description.md                    short purpose and record-relationship note
  date=<YYYY-MM-DD>/id=<id>/
    run.json                        launcher-owned metadata
    logs/{stdout,stderr}.log
    out/                            job outputs only
```

A run ID is `<schema>/<date>/<id>`. The launcher creates the run directory, records command/provenance/inputs in
`run.json`, and finalizes its status. Jobs write only to `$RUN_OUT` (or `{out}`), log to stdout/stderr, and may write
`out/summary.json`.

## Schemas

`schema.py` is the physical schema source of truth: Arrow types, field constraints, table names, Parquet file prefixes,
and DuckDB-view registration. `description.md` explains the purpose and joins; it does not duplicate fields or types.

## Rules

Runs are historical evidence: produce a new run rather than changing a completed one.
