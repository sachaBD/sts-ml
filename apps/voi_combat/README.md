# Python measurement VOI on real combat

The Python selector (`Belief`, `rank_measurements`, `buy`) is the gym's implementation. The bridge runs the native simulator and frozen NN only. The portable interface is a **linear measurement of latent values**, not a simulator-state type.

```
public root -> policy top 4 (keep all if fewer) -> one latent Q per candidate
measurement = Q(move A) - Q(move B) + noise, for each unordered pair
   sample fresh public particle; clone it for both moves
   resolve each public action by key; greedily continue on that world
   stop at next turn's first non-forced decision (evaluate V), or terminal (0/1)
   return the difference; update in Python; re-rank all measurements
play largest posterior mean on the actual fight; reset belief at next decision
```

This is a shallow tree with **fixed** continuations, not a recursive search over interior alternatives. Its Q target is expected endpoint NN value (or terminal win), not unbiased combat win probability. All real-game transfer parameters are initial assumptions, not fitted: policy gaps use temperature .03; prior SD is `min(.06, v, 1-v)`, paired observation SD .08 in the base regime. The default uses a fixed 1× scale; use `--adaptive-scale` to opt into the original 1×/3× mixture. A replay of identical underlying states is not independent NN evidence; randomness here is in the fresh world, not the evaluator. Lambda prices **a complete paired rollout**, not one NN request. The benchmark counts actual uncached requests with identical per-fight cache semantics in all arms.

## Reproduce

From the repository root with the Python environment and native inference dependencies set up. The frozen model and challenge starts below are local benchmark artifacts, not part of the checkout; substitute your own exported checkpoint and fight-start JSONL if needed.

```sh
PY=.venv/bin/python
D=runs/benchmark
B=build/main/apps/voi_combat/voi_combat_bridge
cmake -S . -B build/main -DCMAKE_BUILD_TYPE=Release
cmake --build build/main --target voi_combat_bridge -j6
$PY -m apps.voi_combat.benchmark --bridge "$B" --model "$D/unified-filter/cpu-model" \
    --starts "$D/filter-2400/cal.jsonl" --out runs/benchmark/replay/fights.jsonl \
    --jobs 8 --prices .015 .006 --cap 3
$PY -m apps.voi_combat.report runs/benchmark/replay/fights.jsonl
```

TensorRT smoke: substitute `--gpu --jobs 1 --limit 1 --model "$(cat "$D/unified-filter/model.path")"` and use a fresh output path. The full checkpoint contains fixed-shape GPU models and a CPU overflow model. `gpu_requests` records actual dispatch; the synchronous bridge does not batch across fights. CPU and GPU may make different decisions near ties.

### Calibration run (original fixed-reference selector)

These results predate the all-pairs probe menu.

Frozen iter003; same 360 starts in every arm, 120 per boss; simulator `0300837`. Intervals are paired, boss-stratified percentile bootstraps over decks (4,000 resamples). No capped fights. This is a development comparison, not a fresh held-out estimate. The benchmark refuses to overwrite an existing journal.
