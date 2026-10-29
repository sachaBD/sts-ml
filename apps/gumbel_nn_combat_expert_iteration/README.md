# Gumbel NN combat expert iteration

Thin launcher for `agents/combat/nn/loop.py` on ten fixed Champ decks.

## Run, monitor, stop

From the repo root, with `nn_play` built ([instructions](../../agents/combat/nn/readme.md)).
Use a new `[run].id` and `seed_namespace` in a copy of the preset for a new experiment.

```sh
APP=./apps/gumbel_nn_combat_expert_iteration/run.sh
CONFIG=apps/gumbel_nn_combat_expert_iteration/config/champ_ten.toml
mkdir -p scratch
nohup "$APP" "$CONFIG" > scratch/champ-launch.log 2>&1 &
# Read the run directory from scratch/champ-launch.log, then:
RUN=runs/schema=gumbel_nn_combat_v1/date=YYYY-MM-DD/id=YOUR_ID
watch -n 10 "$APP" status "$RUN"
"$APP" stop "$RUN"
```

The preset **runs until stopped**. No 24-hour timer. Stop finishes the current
collection, training, export and scheduled evaluation, then exits cleanly. Allow
several minutes. `nohup` prevents a closed terminal from killing the run.
Ctrl-C/kill are emergency interruptions and can leave partial output.

## Monitoring agent / future web UI

```sh
"$APP" status "$RUN" --json
```

Text and JSON use the same snapshot: actual running/stopping/stopped/interrupted
state, current phase/progress and its age, latest completed checkpoint, last ten
updates, last three per-deck evaluations and disk free. JSON contains `version: 1`.
Full history stays in `out/training/log.jsonl`; native logs and GPU telemetry stay
beside each stage. There is no general warning/recovery framework.
`run.recovery_retries = 2` opts into a small supervisor for recognised CUDA failures:
fresh process, latest committed checkpoint/optimizer/RNG, partial stage archived,
15-second cooldown, persistent bounded allowance. STOP overrides recovery; unfamiliar
errors and exhausted retries require an agent. Learner stderr is in `logs/learner-*.stderr.log`.
For an inspected interrupted run: `.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.supervisor RUN`.

Without that option, the monitoring agent diagnoses failures and resumes through the existing NN loop:
use `out/command.json` with `--resume-from N`, where N is the latest completed
iteration. Stop any orphan native worker first; move partial `iterNNN` output out
of the way and remove `out/training/STOP` before resuming. Do not append partial
samples. The loop refuses simultaneous runs. See the NN README for recovery details.
Dashboard liveness uses the loop lock, not the original launcher's historical status;
a verified live supervisor's brief cooldown is reported as `recovering`.

External evaluation inputs (`evaluation.starts`, optional `train_starts`, `cohorts`,
`baselines`, `final_starts`) support composition-held-out experiments. Primary and
in-sample validation are separate; only primary results enter the main web curve.
The reserved final starts are copied for later use but **never passed to the loop**.
See `experiments/2026-10-28-champ-generalisation/RUNBOOK.md` for the thousand-deck recipe.

Live browser view of the same snapshot (polls every 2 s, read-only):

```sh
.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.web "$RUN"   # http://127.0.0.1:8740/
```

## Recipe / storage

5,000 fights/update, 100 optimizer steps x 2,048 rows, evaluation initially and every
three updates. Starts untrained; old teacher traces cannot replay safely after
simulator fixes. Historical scores are approximate references on fixed training-deck
monitoring seeds, not controlled algorithm comparisons or unseen-deck tests.

Keep five batches of training samples; load a fresh uniform subset of at most
500,000 decision rows per update using memory-mapped input. Remove expired samples,
old TensorRT caches and committed evaluation sample binaries. Retain checkpoints,
models, small fight summaries and telemetry. No full action traces.

One 5,000-fight batch used 1.24 GiB of replay. A GPU check with five copies and the
500k-row cap used 10.94 GiB peak process RSS (including mapped pages) and 4.18 GiB
peak reserved VRAM, over three optimizer steps. One repeated-batch workload—not
a 24-hour validation. Future fight lengths/token counts can increase usage.
Validation logs: `scratch/nn-24h-readiness/`.

## Fifty-deck experiment

See [`experiments/2026-10-28-champ-fifty/RUNBOOK.md`](../../experiments/2026-10-28-champ-fifty/RUNBOOK.md).
Its config retains the original ten, adds forty capability-checked decks, warms from
that run's latest complete checkpoint, and uses deck-balanced disk-indexed minibatches.
Validation contains100fixedstarts/deck, including the original historical reference
seeds. The dashboard reports Wilson95% win-rate bounds and paired95% differences
against the starting checkpoint, plus historical MCTS20k/PUCT on their500matching
starts only. Original/added cohorts are separate. `validation_share=0.20` budgets
measured validation wall time instead of a fixed evaluation iteration interval.
