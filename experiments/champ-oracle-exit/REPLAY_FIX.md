# r11 oracle replay crash: Liquid Memories fast path

`gen:5650011003206` failed canonical replay at action 4. With preceding start
`gen:5650011003205`, the untouched `pv_worker.ox-e902f9a`, r10 ONNX model,
800 simulations and `--oracle --explore --sample-turns` reproduce the crash.
A fresh worker running the target alone succeeds: the defect is history-dependent.

The target's first three actions are `3,0,2`. At zero energy, Liquid Memories
returns the sole discarded card, Ghostly Armor, to hand slot 2. The simulator's
`chooseDiscardToHandCard(idx, forZeroCost)` ignored its Boolean argument and read
`cardSelectInfo.cardSelectTask` instead. The one-discard automatic path never
opens a selection screen, and that task was not default-initialized. A prior
Liquid Memories task gives cost 0; an INVALID task gives cost 1. Consequently
card action 2 can be legal in the played state and illegal in canonical replay.
This is a simulator defect, not an oracle reuse, sampling, or recording mismatch.

Shared-simulator commit `648229b` honors `forZeroCost` and initializes the
selection task to INVALID. **Behavior change versus the frozen teacher baseline:**
Liquid Memories with exactly one discard now always returns the card at zero
cost. Existing frozen workers remain untouched. No fights are skipped.

Regression checks:
- `sts_liquid_memories_test`: the real potion fast path returns a playable
  zero-cost card for all 25 prior menu tasks; non-free recovery does not inherit
  an old Liquid Memories task.
- `agents.combat.pv.test_oracle_replay`: two starts in
  `testdata/oracle-replay-history.parquet`, in one worker process using the local
  r10 model and the original flags. Completed play verifies full canonical
  replay, final HP, and outcome internally. This reproducible two-start history
  is not claimed to be the exact original concurrent worker assignment.
- Existing `pv_features_test` and `encoding_v4_test`.

Run the model regression with an immutable worker:

```bash
PV_REPLAY_WORKER=/absolute/path/to/new/frozen/pv_worker \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 11 \
  .venv/bin/python -m unittest agents.combat.pv.test_oracle_replay -v
```

Verified on CPU 11: the integration test fails with the original frozen worker
(`invalid action 4`) and passes with the corrected immutable candidate. Both
native checks and the simulator regression pass.

The integration test explicitly skips when the worker or local r10 artifact is
absent. Before comparing future search results, use the same corrected simulator
for both baseline and candidate; old teacher measurements predate this rule fix.
