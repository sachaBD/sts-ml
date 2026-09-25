# slime-v7 execution log

## 2026-09-24: minimal search controls implemented

- User requested a small targeted addition and no tests.
- 24-9-agent implemented optional positive-integer `simulations` and `particles` in `value_play`'s `[run]`, threaded through shared search and effective metadata. Defaults remain 15,000 / 8. No bootstrap configuration changes.
- Early stopping, forced-action budget (min(500, simulation cap)), chunk/batch sizes, and action limits unchanged. Oracle rejects an explicit particle override.
- Implementor reports all targets built cleanly in `build/v7budget` (Release), plus Python syntax compilation. No tests added or run.
- Coordinator inspected the code diff: the requested controls reach search construction, leaf-search budgets, and output metadata. No gameplay validation performed yet.
- `build/valexp/value_play_worker` still needs rebuilding before execution; the isolated build is not the binary snapshotted by value_play.
- At implementation review, no experiment configs had been written and no experiments launched.

## 2026-09-24 21:32 UTC: Stage A launched

- User approved the preregistered four-arm batch: 15k/8, 60k/8, 15k/32, 60k/32; guided rollout, fixed 156 v6 validation fights, random move off, 10 workers, sequential execution.
- Estimated runtime communicated: 25–45 minutes including rebuild; particle-scaling cost uncertain.
- Started via bg_start, job 1: `bash rundecks/slime-v7/stage-a.sh > rundecks/slime-v7/logs/stage-a.log 2>&1`.
- Launcher rebuilds the execution worker. Batch checks R00 decision actions and outcomes against v6 before continuing; execution errors or baseline mismatch stop the batch. A null performance result does not.
- Configs: `configs/slime-v7-r{00,10,01,11}.toml`. No overwrites requested.

## Stage A completed and assessed

- Background job 1 exited 0; all four runs completed. R00 matched all 4,119 v6 decision actions/outcomes before proceeding.
- R00 / R10 / R01 / R11: 121 / 118 / 126 / 125 wins; mean terminal value 0.4274 / 0.4188 / 0.4412 / 0.4400.
- Assessment: no observed gain from 60k at either particle count. 32 particles is promising but unconfirmed; all paired 95% intervals versus R00 include zero.
- Recorded full assessment in `REPORT.md`, numbers in `assessment.json`, reproducible collation in `assess.py`. No further gameplay launched.
- Runtime caveat: host wall-clock timestamps jump during R11; reported monotonic gameplay durations sum to approximately 24 minutes.
