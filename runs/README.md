# runs

`runs/run.py` launches a job into its own run directory. Results live here but are not committed.
`scratch/` has the same layout for disposable work (`--scratch`).

```text
runs/schema=<schema>/date=<YYYY-MM-DD>/id=<id>/
  run.json             written by the launcher: status, command, git revisions, inputs, summary
  logs/{stdout,stderr}.log
  out/                 job outputs only
```

A run ID is `<schema>/<date>/<id>`. Jobs write only to `$RUN_OUT` (or `{out}`), log to stdout/stderr, and may write
`out/summary.json`, which is merged into `run.json`.

The app's schema is `gumbel_nn_combat_v1`. Its `out/` holds:

- `config.toml`, `command.json`: the configuration and launch command.
- `inputs/`: starting fights, evaluation fights and reference scores, and a copy of the `nn_play` binary.
- `training/`: checkpoints, replay samples, ONNX models, results and logs. The dashboard reads these directly.

Runs are historical evidence: produce a new run rather than changing a completed one.
