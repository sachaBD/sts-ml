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

## Value sampling

Set `[training].value_batch = 128` to give combats equal value weight rather than
weighting them by length: one random state per selected combat, resampled each step.
`0` keeps joint policy/value sampling. Policy sampling is unchanged.

Requires `max_train_rows = 0` and replay with recorded combat ranges (new collection).
Changing this setting defines a new experiment. Combat-balanced training value loss
and the existing row/deck-weighted held-out loss measure different distributions.

## Web dashboard

The default **Run health** view shows live execution telemetry from the existing
`/api/actors`, `/api/queues`, and `/api/lanes` endpoints, refreshed every second:

- Actors: inference requests/s and blocked wall-time percentage.
- Queues: requests/s and mean submit-to-batch wait in milliseconds.
- Inference lanes: requests/s, busy wall-time percentage, batch fill, and queue assignment.

Only measured metrics are displayed. Live mode hides the resource cards and mock-only slot counts,
dispatchers, latency percentiles, and lane phase breakdowns. Empty telemetry shows “Not playing / no
telemetry”; failed requests clear prior readings. The API does not expose sample timestamps, so the
view cannot independently detect an old telemetry file. Busy time is not GPU compute utilisation.

**Agent quality** remains a separate view using `/api`. No new production endpoint or sampler is added.
The separate `--mock` preview uses `/api/mock`; only mock mode displays synthetic resource cards and
the richer execution prototype. There is no `/api/health` endpoint.

The normal mock moves through a 48-second loop: balanced, queueing, GPU-starved, and draining.
Five-second rolling summaries refresh once per second. Slot-state proportions, queue average/peak,
dispatcher batch fill, and lane time breakdowns transition smoothly; actor and lane labels show throughput.
All component values are synthetic and visibly labeled DEMO.

A mock backend exercises the UI without run files or a GPU. Mock data is
visibly labeled and the quality tab is disabled:

```sh
.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.web --mock normal --port 8003
# Other scenarios: --mock fallback, --mock stale, --mock missing
```

Tests:

```sh
.venv/bin/python -m unittest apps.gumbel_nn_combat_expert_iteration.web.test_health_mock
node --test apps/gumbel_nn_combat_expert_iteration/web/ui/test_*.cjs
```

## Recovery

`[run].recovery_retries` restarts a run a bounded number of times after recognised CUDA failures; STOP overrides it,
and anything else needs a person. Learner stderr is in `logs/learner-*.stderr.log`.

To resume by hand: stop any orphan `nn_play`, move the partial `iterNNN` aside, remove `out/training/STOP`, then run
`out/command.json` with `--resume-from N` (the latest completed iteration), or
`.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.supervisor RUN`. A resume refuses settings that define
the experiment.

The reserved final fights (`[evaluation].final_fights`) are copied with the run and never played by it.
