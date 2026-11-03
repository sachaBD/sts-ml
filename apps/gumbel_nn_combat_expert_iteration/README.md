# Gumbel NN combat expert iteration

Expert iteration on Champ decks: Gumbel self-play, train on the replay window, export, evaluate.
Settings are defined and checked in [config.py](config.py); inputs are built once with [make_fights.py](make_fights.py).

## Run, monitor, stop

From the repo root, with `nn_play` built ([setup](../../README.md#setup)).
Use a new `[run].id` and `seed_namespace` in a copy of the preset for a new experiment.

```sh
APP=./apps/gumbel_nn_combat_expert_iteration/run.sh
CONFIG=apps/gumbel_nn_combat_expert_iteration/config/champ_ten.toml
mkdir -p scratch
nohup "$APP" "$CONFIG" > scratch/champ-launch.log 2>&1 &
# Read the run directory from scratch/champ-launch.log, then:
RUN=runs/schema=gumbel_nn_combat_v1/date=YYYY-MM-DD/id=YOUR_ID
watch -n 10 "$APP" status "$RUN"            # --json for agents
.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.web "$RUN"   # http://127.0.0.1:8740/
"$APP" stop "$RUN"
```

The preset runs until stopped. Stop finishes the current iteration and its scheduled evaluation, then exits
cleanly; allow several minutes. Ctrl-C/kill can leave partial output.

## Web dashboard

The default **Run health** view contains a resource-headroom frame and an execution-diagram prototype: host CPU busy
and per-logical-CPU activity, GPU device 0 busy time and VRAM occupancy, RAM used/available, and
run-filesystem disk used/available. Machine sampling is approximately every two seconds. Missing
or stale measurements are labeled, not displayed as zero usage. These are machine-wide counters,
including other workloads; GPU kernel-active time is not percent of peak compute throughput.
Available RAM includes reclaimable cache; available disk space excludes reserved filesystem space.

**Agent quality** remains a separate view. The health sampler and `GET /api/health` do not call the
expensive per-deck scoring endpoint. The sampler retains phase-local history for up to 30 minutes,
in memory since dashboard launch, but the resource frame deliberately only shows current usage.
No training code or native worker instrumentation is changed by dashboard sampling.

The **Execution** diagram shows individual actors, a shared queue, dispatchers, and GPU lanes.
The component-state visual is currently a mock prototype: it is not yet connected to the production
telemetry API, so the real-run diagram marks component telemetry unavailable rather than inventing
activity. Existing resource metrics remain live. The `/api/actors`, `/api/queues`, and `/api/lanes`
endpoints remain available; connecting the diagram to them is a separate next step.

The normal mock moves through a 48-second loop: balanced, queueing, GPU-starved, and draining.
Five-second rolling summaries refresh once per second. Slot-state proportions, queue average/peak,
dispatcher batch fill, and lane time breakdowns transition smoothly; actor and lane labels show throughput.
All component values are synthetic and visibly labeled DEMO.

A deterministic mock backend exercises the same API/UI without run files or a GPU. Mock data is
visibly labeled and the quality tab is disabled:

```sh
.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.web --mock normal --port 8003
# Other scenarios: --mock fallback, --mock stale, --mock missing
```

Tests:

```sh
.venv/bin/python -m unittest apps.gumbel_nn_combat_expert_iteration.test_health
node --test apps/gumbel_nn_combat_expert_iteration/ui/test_health_ui.cjs apps/gumbel_nn_combat_expert_iteration/ui/test_web_ui.cjs
```

## Recovery

`[run].recovery_retries` restarts a run a bounded number of times after recognised CUDA failures; STOP overrides it,
and anything else needs a person. Learner stderr is in `logs/learner-*.stderr.log`.

To resume by hand: stop any orphan `nn_play`, move the partial `iterNNN` aside, remove `out/training/STOP`, then run
`out/command.json` with `--resume-from N` (the latest completed iteration), or
`.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.supervisor RUN`. A resume refuses settings that define
the experiment.

The reserved final fights (`[evaluation].final_fights`) are copied with the run and never played by it.
