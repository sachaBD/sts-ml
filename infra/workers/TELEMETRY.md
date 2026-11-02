# Runtime telemetry

`run_actors` accepts an optional `RuntimeObserver`. Without one, slot, queue and lane instrumentation
is disabled. An observer runs on the existing supervisor approximately once per second and once at
termination. Observer exceptions disable reporting; they do not abort episodes or change routing.
Observers must return promptly. No worker thread performs serialization or filesystem I/O.

`nn_play` enables reporting when `STSRL_RUNTIME_TELEMETRY` names an output file. The Python play
launcher sets it to the stage's `runtime.json`. Older binaries ignore the environment variable.
Each report atomically replaces that file via a same-directory temporary file and rename. Reporting
failure is logged and disabled, not treated as a training failure. Nothing changes checkpoints,
model formats, predictions, routing decisions, or game journals.

## Snapshot protocol, schema_version 1

All counters are cumulative within `worker_instance_id`; IDs are stable within that instance.
The writer combines PID and a steady-clock start value for the instance ID. A new process requires
a new baseline. `sequence` increases for each report. `captured_at` is Unix seconds for freshness;
`elapsed_seconds` is steady-clock time for rate denominators. Component locks are sampled in turn,
not as a globally atomic stop-the-world snapshot.

- `actors`: ID, slot capacity, submitted requests, completed games, cumulative `working_seconds`,
  and `slot_seconds` for `ready`, `awaiting_inference`, and `empty`.
  Ready includes a slot currently advancing. Slot-seconds are time integrals, not counts inferred
  from periodic state samples. Working is host wall time excluding the Actor's inference condition
  wait and time before start/after completion; it is **not thread CPU time**.
- `queues`: one queue per backend, with current depth, cumulative depth-seconds, oldest pending
  request age, and a cumulative wait histogram. Queue wait is enqueue to dequeue, after input
  preparation, excluding time in a gathered batch or inside the backend. Histogram bins are
  non-cumulative, bounded by `upper_bounds_seconds`, with one final unbounded overflow bin.
  `peak_depth_interval` covers the interval since the previous report, including its starting
  depth. A reader missing sequences cannot recover the exact peak for the missing intervals.
- `dispatchers`: stable ID and backend ID, completed batches, request rows, and gathered capacity
  rows. Gathered capacity uses that queue's cap, not a GPU engine's padded shape.
- `backends`: route ID, kind, max batch, token/attention capacity (zero means unspecified),
  completed oversized CPU rows, completed precision retries, and per-GPU-lane counters.
- `lanes`: completed batches, real GPU rows, padded capacity rows, and exclusive host-observed
  `time_seconds` in idle, packing, H2D, inference API, D2H/unpacking, and CPU retry.
  Active intervals are included while a call is in flight. Occupancy includes host work and
  synchronous driver calls; it is **not GPU compute efficiency**. Oversized CPU fallback happens
  outside a GPU lane. Precision-retried rows remain GPU rows and must not add requests to the
  denominator a second time.

Readers subtract counters across snapshots of the same instance. Windows must report actual
coverage and missing sequences, pool histogram counts before computing quantile bounds, and
mark absent/stale measurements rather than substitute zero. Polling `runtime.json` does not itself
produce a new observation; deduplicate by instance ID and sequence.
