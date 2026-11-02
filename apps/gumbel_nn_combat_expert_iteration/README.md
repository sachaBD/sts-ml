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

## Recovery

`[run].recovery_retries` restarts a run a bounded number of times after recognised CUDA failures; STOP overrides it,
and anything else needs a person. Learner stderr is in `logs/learner-*.stderr.log`.

To resume by hand: stop any orphan `nn_play`, move the partial `iterNNN` aside, remove `out/training/STOP`, then run
`out/command.json` with `--resume-from N` (the latest completed iteration), or
`.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.supervisor RUN`. A resume refuses settings that define
the experiment.

The reserved final fights (`[evaluation].final_fights`) are copied with the run and never played by it.

## Optional wide inference route

For an entity model, these additions retain two narrow lanes and add one wide lane:

```toml
[model] # additions to the existing model section
# gpu_tokens = 64
# gpu_attended = 24
# kind = "entity"
gpu_wide_tokens = 128
gpu_wide_attended = 64

[workers] # additions to the existing workers section
# gpu_batch = 512
gpu_lanes = 2
gpu_wide_lanes = 1
gpu_wide_batch = 32
```

`gpu_wide_tokens = 0` (the default) disables the wide route. Existing configs and checkpoint formats
remain valid. The exporter writes `model-gpu-wide.onnx` from the same weights; it has its own graph
cache, so equal narrow/wide batch sizes cannot reuse the wrong topology. Shapes and lane counts are
fixed for a run, like the existing deployment settings, and may not change on resume.

The native worker routes before inference using total tokens and unpooled attention positions supplied
by the combat encoder. Each route has separate queues, batch limits and dedicated lanes. Oversized
states use the full CPU model on the widest route; numerical GPU failures still retry on CPU. Neither
path truncates states/actions, and neither occupies the narrow inference servers when a wide route is
configured. Extremely large states can still delay the wide route or hit the existing request timeout.

Each lane owns a stream, pinned/device buffers and model sessions; all CUDA graphs are captured before
worker threads start. Lanes share GPU resources rather than reserve compute partitions. More lanes
increase GPU memory and server-thread costs. The old synchronization policy is unchanged.

`gpu_stats` remains the aggregate for dashboards. `gpu_route_stats` adds per-route GPU/CPU counters;
`inference_route_stats` reports per-route completed requests, batches, peak queued requests and mean
queue wait at shutdown. Route 0 is narrow, route 1 wide. Queue peaks exclude requests already being
computed. Timings summed across lanes are not GPU utilization.

Short synthetic backend screens on the RTX 3060 measured 103.7k / 120.7k / 120.5k evaluations/s for
1 narrow + 1 wide / 2 narrow + 1 wide / 4 narrow + 1 wide (one approximately five-second trial each,
99% narrow / 1% wide, variability unmeasured). These informed the optional configuration, not a promise
of gameplay throughput or coverage beyond 128 total / 64 retained tokens. Real-workload checks were
explicitly not performed for this change.
