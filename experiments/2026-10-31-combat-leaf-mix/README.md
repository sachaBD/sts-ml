# Checkpoint-132 leaf-mix sweep

Completed: see [results](RESULTS.md). Compared **75% NN / 25% rollout** and **100% NN** on the
same 500 starts (50 per training deck), reusing cached NN50 results. Hold search at
256 simulations, eight particles, zero noise. Run sequentially, once per variant.

`protocol.toml` pins the original run, checkpoint, ONNX exports, worker and evaluation
files by SHA-256. Cached wins / 500: **NN50 288, PUCT 237, MCTS20k 175**.

From this worktree:

```sh
PY=/home/sborowsk/project/sts/combat_rl/.venv/bin/python
EXP=experiments/2026-10-31-combat-leaf-mix
$PY "$EXP/benchmark.py"  # read-only preflight; no GPU or fights
# Only after execution approval:
$PY "$EXP/benchmark.py" --run runs/benchmark/leaf-mix-132-ATTEMPT
```

The output directory must be new and source committed. Save protocol, revision, exact
commands, outcomes, logs and whole-process wall time. `wall_seconds.txt` is written
only after a variant returns all 500 valid results. No retries, resume or cleanup.
Per-deck results and paired differences are recorded in `RESULTS.md`.

Caveats: these are training-monitoring starts, not held out. Historical PUCT/MCTS use
older simulators. Use the frozen worker, not main (PR #18 changed blending/RNG);
at mix 0 its search RNG consumption changes too. Its exact dirty source/build was not
archived. One timing observation/variant cannot establish timing variability or a
controlled speedup against the cached baseline. Both native runs completed successfully.

## Follow-up: NN-valued chance frontier (not implemented)

At each stochastic boundary, sample N outcomes through to valid next decision states,
evaluate the NN at each (terminal outcomes use 0/1), average and cache the value.
Do not expand below that boundary or resample it on later visits. Compare N and wall time
against particle-tree search; repeated cached backups are not new samples. This tests
search depth/particle sensitivity, not by itself whether strategy fusion exists.
