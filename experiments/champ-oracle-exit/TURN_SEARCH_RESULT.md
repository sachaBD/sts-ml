# Turn search v1 — evaluation-only checkpoint

Implemented against fixed simulator `648229b`; `search.hpp` and existing per-action
search behavior are unchanged. The census key implementation was moved unchanged
to `agents/combat/pv/turn_state_key.hpp` (full binary keys, no canonical sorting).

## Run

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 11 \
  .venv/bin/python apps/pv/play.py --agent pv --oracle --turn-search \
  --model runs/schema=combat_v4/date=2026-10-04/id=champ-ox-c-r09/model/model.onnx \
  --sims 64 --workers 1 --worker /absolute/path/to/frozen/pv_worker \
  --starts STARTS.parquet --out OUT
```

`--sims` is E in this mode. Requires `--oracle`; exploration, early-turn sampling,
policy-only inference and rollout mixing are rejected (evaluation-only v1).
Defaults: root 2048 children, non-root 512, 20,000 sequences, 1 second per expansion,
512 actions/path, conservative 256 MiB tree/workspace accounting, T=10, c=1.25.
Worker-only overrides: `--turn-root-max-children`, `--turn-max-children` (non-root),
`--turn-max-sequences`, `--turn-max-seconds`.

DFS completes mandatory choices during END_TURN resolution before keying the next
player-normal decision. Children retain a sequence, full key and value plus tree
bookkeeping, never a BattleContext; expansion materializes states by checked replay.
Distinct nonterminal children are evaluated in **bounded batches of at most 32**,
not one unbounded call (approved clarification). Values are clamped to [0,100];
terminal win/loss values are 100/0. Every descent/expansion attempt consumes E,
including terminal and previously capped leaf backups. E=1 expands a fresh root
exhaustively and chooses its highest-valued child, with no child pseudovisits.

Root cap → ordinary oracle per-action PUCT, 800 simulations per decision for the
whole turn. Non-root cap → retain its own V as a leaf. No partial expansion is
published. Pending callbacks at an undecided keyed boundary abort. Time checks
are cooperative; elapsed time and overshoot are recorded. Subtree reuse is exact-key
checked and bounded.

`turn_stats-*.parquet` contains per-turn elapsed seconds, attempt count, network
calls, evaluated states, root children, tree max depth, mean leaf depth, fallback
reason, overshoot and reuse. Mean leaf depth is unweighted over retained tree
leaves; max depth counts macro edges from the current root. Ordinary search and
per-action stats rows are emitted **only on fallback turns**; successful macro
turns invent no policy targets.

## Verification

- Native `pv_turn_search_test`: stored sequence/key replay, E=1 optimum, win/loss,
  root sequence/child/action/time/memory caps, separate root/non-root child caps,
  non-root all-or-nothing leaf retention, exact reuse, END_TURN Codex choices,
  callback rejection and value clamping.
- Existing `pv_features_test`, `encoding_v4_test` pass.
- Worker tests: no-flag stable JSON byte-identical to immutable pre-turn fixed
  worker after removing timing; missing oracle rejected; fallback and no invented
  macro policy rows checked.
- r10 two-fight Liquid Memories history regression still passes.
- All builds/runs pinned to CPU 11; builds `-j1`, one ONNX worker. Existing frozen
  binaries were not overwritten.

## Final smoke: root 2048 / non-root 512, E=64

Artifact directory:
`runs/schema=combat_v4/date=2026-10-04/id=champ-ox-turn-search-e64-root2048-smoke/`.
Frozen worker: `build/frozen/pv_worker.turn-search-v1-58c5c4ab8d3d`
(SHA-256 `58c5c4ab8d3dbbd6868a22bb86b467b08d3686fd8e0f368206b9b5368d83a6b0`).
Starts are the first 20 rows of the nodome bench; model is r09. These are **20
convenience-slice fights, not a powered/randomized comparison**. No corrected-simulator
paired per-action baseline or full 409-fight comparison was run.

| Metric | Result |
|---|---:|
| Wins / completed | **5 / 20 (25%)** |
| Fight caps | 0 |
| Wall seconds / fight | **4.47** (89.41 seconds total) |
| Decision seconds / fight | 4.16 |
| Per-fight decision seconds: median / p90 | 3.72 / 6.26 |
| Per-fight decision seconds: observed range | 0.00031–17.14 (n=20; not uncertainty bounds) |
| Turn decisions / successful macro searches | 156 / 134 |
| Root fallback | **22 / 156 (14.10%)** |
| Child-cap fallback | 20 / 156 (12.82%) |
| Sequence-cap fallback | 2 / 156 (1.28%) |
| Other fallback reasons / time overshoots | 0 / 0 |
| Successful-turn tree max depth: mean / p90 / max | **2.02 / 3 / 6 turns** (n=134) |
| Successful-turn mean leaf depth: mean / p90 | 1.68 / 2.66 turns (n=134) |
| All-turn tree max depth: mean / p90 | 1.74 / 3 turns (fallback depth=0) |
| Network calls / evaluated states (including fallback) | 15,949 / 464,454 |

For context only, the earlier 512-root smoke had 58/155 fallback turns (37.42%)
and the same 5/20 wins. Its two runs took 47.98 and 79.81 wall seconds; timing
variability is substantial and its cause was not measured. Do not treat the
single-run 2048-vs-512 timing difference as a controlled speed comparison.

Interpretation: larger root caps reduce fallback, but this design still searches
only a few turns deep on this slice. Win-rate uncertainty/generalization to the
409 bench is unknown; these results do **not** establish that depth is irrelevant.
The old 41.3% teacher baseline also predates the rare Liquid Memories simulator
behavior correction and is not a matched comparator.

Checkpoint complete: tests + 20-fight smoke, then commit and stop. No v2/training
integration or full-bench experiment was added.
