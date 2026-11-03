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

## Independent value sampling

`[training].value_batch = 128` keeps the existing policy replay batch, but trains value on
128 independently selected distinct combats per optimizer step, one random retained state each.
Combats and states are resampled every step. Both losses are means with coefficient one;
the shared trunk receives both gradients before one joint clip and optimizer update.
This is separate supervision, not separate networks or a frozen policy representation.

The default `value_batch = 0` preserves joint policy/value training on the same states.
Value sampling respects the replay window and `max_train_rows`: combats without retained states
are excluded. A positive value batch must not exceed the number of eligible combats;
otherwise training fails before updating the model. `balanced_replay` continues to control
policy sampling only when independent value sampling is enabled.

Changing `value_batch` defines a new experiment, not a resumable setting. Architecture,
checkpoint layout, and inference are unchanged. Training value loss then measures the
combat-balanced value samples, while fresh/held-out scoring retains the original replay
sampling distribution; the two logged losses are not directly comparable.

## Recovery

`[run].recovery_retries` restarts a run a bounded number of times after recognised CUDA failures; STOP overrides it,
and anything else needs a person. Learner stderr is in `logs/learner-*.stderr.log`.

To resume by hand: stop any orphan `nn_play`, move the partial `iterNNN` aside, remove `out/training/STOP`, then run
`out/command.json` with `--resume-from N` (the latest completed iteration), or
`.venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.supervisor RUN`. A resume refuses settings that define
the experiment.

The reserved final fights (`[evaluation].final_fights`) are copied with the run and never played by it.
