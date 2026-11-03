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

The **Run health** resource frame and **Execution** diagram are mock-only previews in this PR.
There is no live health sampler or `/api/health` endpoint. The preview uses `/api/mock`, available
only when launched with `--mock`. Live-run wiring is deliberately deferred; this view will not
work for real runs yet. **Agent quality** remains a separate view.

The existing `/api`, `/api/actors`, `/api/queues`, and `/api/lanes` routes remain available for real runs.

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
node --test apps/gumbel_nn_combat_expert_iteration/web/ui/test_health_ui.cjs apps/gumbel_nn_combat_expert_iteration/web/ui/test_web_ui.cjs
```


## Recovery

`[run].recovery_retries` restarts a run a bounded number of times after recognised CUDA failures; STOP overrides it,
and anything else needs a person. Learner stderr is in `logs/learner-*.stderr.log`.

To resume by hand: stop any orphan `nn_play`, move the partial `iterNNN` aside, remove `out/training/STOP`, then run
`out/command.json` with `--resume-from N` (the latest completed iteration), or
`.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.supervisor RUN`. A resume refuses settings that define
the experiment.

The reserved final fights (`[evaluation].final_fights`) are copied with the run and never played by it.
