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

## Evaluation sets

Each set explicitly owns its fights, search, and schedule; nothing is inherited from self-play.
See the preset for a complete config.

```toml
[[evaluation.sets]]
name = "validation"
fights = "path/to/validation.jsonl"
sims = 64
gumbel_scale = 0.0
rollout_mix = 0.0
particles = 8
at = [0]
every = 3
```

`at` adds iterations; `every = 0` disables periodic evaluation. `[evaluation] sets = []` disables all sets.
Schedules and ordering may change on resume; names, fights, and search settings may not.
Inputs are frozen at `inputs/evaluation/NAME.jsonl`; results go to `training/test/iterNNN/NAME/`,
with one completion log row per set. Recovery retries only uncommitted sets.

This replaces `evaluation.fights` / `in_sample_fights`; old runs are not automatically migrated.
Named-set dashboard quality views are a follow-up; health/status remain available.

## Adaptive deck sampling

```toml
[starts] # add to existing starts settings
history_iterations = 10
win_rate_prior_strength = 1.0
sampling_concentration = 0.5 # 0: uniform (default); 1: maximum uncertainty only
```

Before each self-play iteration, use this experiment's preceding `history_iterations`
completed batches (or all available). For each deck, with prior strength `s`:

```text
p = (wins + s) / (attempts + 2*s)
weight = [4*p*(1-p)] ** [concentration/(1-concentration)]
probability = weight/sum(weights)
```

The window must be positive; `s` is a finite positive pseudocount per outcome.
Unobserved decks have maximum weight; no history means uniform sampling.
Concentration 0 is uniform; 0.5 uses uncertainty weights directly; 1 selects only
maximum-uncertainty decks (ties uniform).
Defaults are window 10, prior 1.0, concentration 0.0; disabled sampling preserves the
original random stream. All three settings are fixed on resume; warm starts have no history.

Counts, weights, probabilities and window settings are saved atomically before collection
in `training/sampling/iterNNN.json`, outside interrupted iteration archives, and reused on
recovery. No external weight file is needed. State variation, replay and evaluation are unchanged.

## Value sampling

Set `[training].value_batch = 128` to give combats equal value weight rather than
weighting them by length: one random state per selected combat, resampled each step.
`0` keeps joint policy/value sampling. Policy sampling is unchanged.

Requires `max_train_rows = 0` and replay with recorded combat ranges (new collection).
Changing this setting defines a new experiment. Combat-balanced training value loss
and the existing row/deck-weighted held-out loss measure different distributions.

## Value head

`[model].value = "outcome"` predicts the combat's outcome as 20 bins: losses by enemy HP left, then wins
by player HP fraction. Search still receives the win probability (the win bins' mass), so export and native
inference are unchanged. `value_loss` is then the outcome cross-entropy; `win_loss` is the win BCE in either
mode and is the comparable number across runs. Requires sample format 2 (cold start).

`[model].auxiliary = true` adds training-only heads, weighted 0.25: HP lost and enemy HP lost by the start
of our next turn (`round_loss`, on policy batches) and turns remaining (`turns_loss`, on value batches).

## Learning rate schedule

`[training]` `lr_half_life = 30` and `lr_final = 0.0001`: iteration `i` trains at
`lr_final + (lr - lr_final) * 0.5 ** (i / lr_half_life)`. `0` (default) keeps `lr` constant.

## Web dashboard

The default **Run health** view shows live execution telemetry from the existing
`/api/actors`, `/api/queues`, and `/api/lanes` endpoints, refreshed every second:

- Actors: inference requests/s and blocked wall-time percentage.
- Queues: sampled depth, peak since worker start, requests/s and mean submit-to-batch wait.
  Bars use a labelled display scale, not queue capacity. Older workers show unknown depth/peak.
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
