# Exact-turn enumeration measurement

`pv_worker turns` reads recorded fights as JSON lines **over a pipe only**. `apps/pv/turns.py` supplies the first
50 rows from the selected fights Parquets, writes `turns.parquet`, and summarizes it with DuckDB (512 MB, one thread).
Full fights are validated with `combat_v4::replay`; turn starts are the first recorded decision at each new zero-based
`BattleContext.turn`, rebuilt from the same start/action prefix.

DFS enumerates every simulator-legal action (no PV semantic action deduplication), stopping a sequence when
END_TURN executes, a card automatically advances the turn, or combat terminates. END_TURN execution includes the
enemy turn and next-turn draw up to the next simulator decision. Enumeration caps: 1,000,000 completed sequences,
512 actions on a path (zero-cost cycles), or an estimated 1 GB of key-set storage. Cap reason is recorded; capped
turns contain partial counts. Time includes state copying, enumeration, keys, and differential checks.

## Keys and boundaries

No complete simulator serializer exists. The measurement serializes every BattleContext scalar/RNG, all Player
fields and sorted status-map entries, all inline monster/power/counter/scratch data, full card attributes including
unique IDs, and actual ordered pile contents. Trivial nested structs have GCC padding clearing; BattleContext and
Player are serialized member by member. Debug counters and some inactive inline fields are retained, so counts
are conservative upper estimates of future-equivalent states, **not guaranteed minimal equivalence classes**.

Exclusions: allocator/container pointers; inactive hand slots; empty action/card queues (including stale callbacks,
indices and slots); limbo/current card-queue item when the card queue is empty and input is PLAYER_NORMAL; static
simulator debugging globals. No active future-effect field is intentionally excluded. Pending callbacks at any
**undecided** endpoint abort the measurement because `std::function` captures cannot be serialized generically.
Terminal sequences need no key and are excluded from distinct counts; their wins/losses are counted separately.

The exact key preserves hand/draw/discard/exhaust ordering. The canonical key sorts hand/discard/exhaust as
multisets, retaining ordered draw. Canonical keys are candidate buckets, **not equivalence classes**: fixed-RNG
random hand targets, bulk triggers and return order can make them lossy. No simulator rule is changed.

Sets retain the first 128 SHA-256 bits rather than full strings (OpenSSL Crypto). This is probabilistic identity:
under an ideal uniform hash, the birthday collision probability at 1e6 entries is approximately 1.5e-27 per set.
The one-GB backstop conservatively accounts for node/bucket overhead as well as digest payloads.

## Checks and differential sample

Each input fight runs three full-key sanity checks on a stripped, non-interacting fixture derived from its real
RNG/monster context: identical sequence replay twice; commuting Block/Strength potion consumption; and
Defend→Strike versus Strike→Defend followed by END_TURN. The last pair must merge under the canonical key but
remain different under the exact key. The fixture keeps enough draw cards to avoid an immediate reshuffle.

At most 200 pairs merged by the canonical key but not the exact key are tested. Sampling is deterministic: retain
the first 64 candidate buckets per turn and check at most five pairs per turn. From endpoint A, take the first
legal action repeatedly to END_TURN. Replay on B, relocating card plays by unique ID/target; potion slots/targets
and selection bits are unchanged. Illegal replay and 512-action continuation caps are separate outcomes. Terminal
leaves compare their known win/loss outcome, not terminal keys. Divergence is reported among completed paired
continuations. This is a non-random, within-turn-correlated diagnostic, not an unbiased estimate for all buckets.

Example (single core; freeze a built worker before launch):

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 11 .venv/bin/python apps/pv/turns.py \
  --fights runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r09/bench-oracle \
  --worker build/frozen/pv_worker.turns-COMMIT --n 50 \
  --out runs/schema=combat_v4/date=2026-10-04/id=champ-ox-turns-c-r09/out
```
