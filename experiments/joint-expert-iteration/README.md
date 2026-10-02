# Joint combat / overworld expert iteration

- [PLAN.md](PLAN.md): approved plan and scope.
- [RUNBOOK.md](RUNBOOK.md): timestamped launches, tests, gates and failures.
- [REPORT.md](REPORT.md): updated during execution, finalized at completion.
- `driver.py`: bounded sequential controller; frozen policies per batch, paired gates, fresh final confirmation.

Run from repository root with `PYTHONPATH=.`. The driver is launched through `runs.run`; its checkpoint/state and binary snapshot are in that managed run's `out/`. Each collection, evaluation and model training stage has its own managed run and logs. Do not overwrite historical runs. `--no-compact` is deliberate for current multi-table files.

Resume the driver using the same `--out` to skip completed child stages; incomplete child stages stop with a diagnostic rather than silently dropping episodes. The controller uses a hard wall deadline and terminates the active stage process group if it expires. Inspect partial outputs before resuming a failed stage.

Canonical data: `combat_v4` initial state + exact executed actions; separate results/search plus disposable legacy-compatible training cache. Unsupported starting callback boundaries have result-only records and `replay_error`, not invented snapshots/labels. `overworld_v1` stores actual public macro-state/choice traces, linked fight IDs and separate value/exploration annotations. It does not claim full GameContext replay.
