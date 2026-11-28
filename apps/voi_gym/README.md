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
