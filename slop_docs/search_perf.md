# Search performance (branch `search-perf`, here and in sts_lightspeed)

Goal: more data-generation throughput (teacher fights per core-hour) at little or no cost to outcomes.
Measured on one or two cores of a shared machine, so timings are CPU time (`CLOCK_THREAD_CPUTIME_ID`), and
every comparison runs the baseline and the variant side by side.

## 1. Exact speedups (default on, output unchanged)

Every row these write is byte-for-byte what the old code wrote. Checks:
- `bench_search` traces: every row of 3 Slime Boss fights, value-net teacher at 20k.
- `fight_resample_worker` `result.msgpack`: 15/15 identical on real act1-a20-8 decks with the value-net
  teacher (15k), plus one fight with the guided-rollout teacher.
- `tests/search_perf_test.cpp`: 12k states, including search leaves.

| change | where | effect |
|---|---|---|
| value-net leaves encoded by `encode_state` (skips the legal-action tokens and their `printDesc` strings, which the net never reads) | combat/environment.cpp, teacher_leaves.cpp | biggest win: encode went from ~15 to ~2 µs per leaf |
| tree node keys: `observationKey` (same fields and equality as `publicObservation`, but one 64-bit mix per field instead of byte-wise FNV). The seeds still use `publicObservation` | PublicBeliefCombatSearch | ~12% |
| value net: caches the monster / interaction MLP outputs (bit-exact keys), accumulates each layer in registers, no allocation per state, bit-keyed card cache | models/value_net.cpp | eval went from ~12 to ~2 µs per leaf |
| `ActionQueue`: popped / cleared slots are emptied. Stale `std::function`s used to be copied with every `BattleContext` | sts_lightspeed ActionQueue.h | ~9% on guided rollouts |
| pending leaf requests reuse their map nodes; `select` loop invariants hoisted | PublicBeliefCombatSearch | small |
| `encode_state` skips `card_meta` for non-hand cards | environment.cpp | small |

**Result (CPU time, new / old):**
- **Value-net teacher:** 0.38 on 15 real-deck fights (15k simulations), 0.37–0.40 on Slime Boss scenario fights (20k). That is **~2.6× throughput**.
- **Guided-rollout teacher:** 0.72–0.85.

Not changed: the game engine's own action execution (~30% of the remaining value-net time), and the net's
arithmetic (`value_net.cpp` is still built with `-march=native`).

## 2. Opt-in search variants (`teacher::SearchTweaks`, default off)

These are set per run as teacher settings. `value_play` reads them from the `[run]` keys `stop_factor` and
`merge_identical_cards`. Any worker can accept them with one line in its teacher-settings loop:
`else if (stsrl::teacher::set_tweak(key, value)) {}`. `settings()` records non-default values.

- **`stop_factor` (0, 1]:** early stop once N_best − N_second > f × simulations left (1 = the exact rule).
- **`merge_identical_cards`:** identical cards in different hand slots become one edge. Before, only adjacent
  duplicates were merged, so e.g. two Strikes split one move's visits, which also blocks early stop.

### How they were measured

`bench_search regret`, value-net teacher at 20k, on 3 × 270 multi-move states from independent seeds:
1. Each state gets a 100k-simulation reference search, giving Q_ref for each move.
2. regret = max Q_ref − Q_ref(chosen), shown ×1e-3.
3. Ref-merged rows used a merged reference (rounds 1–2); ref-unmerged rows used an unmerged reference (round 3).

| variant | simulations | same move as baseline | regret vs baseline (95% CI) | root_value shift |
|---|---|---|---|---|
| reseeded baseline (noise) | 1.00 | 256/268 | −1.2 (−2.7, +0.4) | ±0.0007 (abs) |
| stop_factor 0.5 | 0.91 | 268/268 | 0 | −0.0026 |
| stop_factor 0.25 | 0.80 | 273/275 | +1.1 (−0.5, +2.8) | −0.0055 |
| stop_factor 0.1 | 0.64 | 264/268 | +0.8 (−0.3, +1.9) (ref unmerged) | −0.0083 |
| merge + stop 0.5 | 0.84–0.86 | ~88% | −3.4 / −5.4 (ref merged, significant); +0.1 (−3.1, +3.2) (ref unmerged) | −0.004 to −0.005 |
| 10k simulations (for scale) | 0.52 | 239/268 | +4.8 (+1.9, +7.7) | −0.0046 |

- **Early stop keeps the moves but shifts the labels.** It barely changes the moves, but it lowers
  `root_value` (the mean over root visits: fewer simulations leave more exploration visits in it). That is a
  small but systematic shift in the training target. This is why it is opt-in.
- **`stop_factor` 0.5 is a near-free ~10%.** 0.25 is ~20%.
- **Merging is at least neutral for quality.** It is clearly better only when judged by a merged reference,
  so part of that gain may be reference bias. It saves ~5% of simulations by itself.

## 3. Tried and rejected

- **Tree reuse (PR #1, `reuse`) with the value-net teacher:** regret +2.6 (−1.2, +6.4), and simulations only
  0.94. Not worth it here.
- **Transpositions within a turn** (a DAG keyed by turn start plus observation): no gain in the tree size and
  no regret gain. Reverted.
- **Hand-order invariance at chance nodes:** not pursued. With 8 fixed particles, the next turn's hand is
  almost never reused anyway.
- **Fewer simulations:** regret rises clearly at 10k and 5k. The budget is binding.

## Tools

`apps/bench_search.cpp` (CPU time; `GUIDED=1` selects the guided-rollout teacher). Modes:
- `play WEIGHTS FIRST COUNT [SIMS]`: per-fight time. Set `TRACE=file` to dump every row for exactness diffs.
- `phases`: split into tree, encode, eval and backup. A `-DPBCS_PROFILE` build adds the tree's per-phase cycles.
- `evalbench`: encode and eval microseconds per leaf.
- `paired`: one variant vs baseline along the baseline's trajectory.
- `regret`: variants vs a reference search. `ARMS=base,merge+stop=0.5,...`, `REF_SIMS`, `REF_MERGE`.
