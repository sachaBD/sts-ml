# Measurement VOI gym

```sh
PY=.venv/bin/python
$PY -m apps.voi_gym.explorer --host 0.0.0.0 --port 8006
$PY -m apps.voi_gym --seeds 128 --budget 4 --prices .006 .001
$PY -m unittest agents.combat.search.voi_gym.test_model apps.voi_gym.test_session apps.voi_gym.test_explorer apps.voi_combat.test_benchmark
```

The toy and native combat benchmark share the root selector. `Q` scores public
states and returns a value and policy prior; `S` takes one action in a sampled
world. The duel uses a deterministic heuristic in place of a network.

Every unordered pair of shortlisted moves is eligible. Each purchased pair
compares its two moves on a shared world. Both continuations are greedy and finish at a terminal or next-turn
state. The replay counts cache misses at `Q` and all transitions through `S`,
including the forced resolve action. Sampling a world is not an `S` call.

The default belief uses a fixed 1× scale. Select **Adaptive 1× / 3×** in the
explorer or pass `--adaptive-scale` to opt into the original mixture.

## Native fight recordings

Build `voi_combat_bridge` and use a local exported CPU model and starts JSONL
(see `apps/voi_combat/README.md`). Then record and serve them alongside the gym:

```sh
$PY -m apps.voi_gym.record_combat --bridge build/main/apps/voi_combat/voi_combat_bridge \
  --model "$MODEL_DIR" --starts "$STARTS_JSONL" --out runs/fight-replays.jsonl --limit 3
$PY -m apps.voi_gym.explorer --host 0.0.0.0 --port 8006 --recordings runs/fight-replays.jsonl
```

Native counters are local to the selected decision. NN evaluations count
cache misses (including the root); rollout actions count every simulated action,
including forced actions, but not actions taken in the actual fight. Recordings
show the settings used at capture time. Pass `--help` to the recorder for
shortlist, noise, prior, uncertainty, price, and budget options.

## Live native replay

Add `--bridge build/main/apps/voi_combat/voi_combat_bridge --model "$MODEL_DIR"
--starts "$STARTS_JSONL"` to the explorer command, using the same model and
starts as the recording. Rerun a selected decision or play to the next action
or turn. Controls apply to the current branch; choosing a recorded decision
again resets it. Each request restores the recorded command prefix and replays
committed branch actions, so deeper positions may take longer. The displayed
counters belong to the selected decision, not that reconstruction work.
