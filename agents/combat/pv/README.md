# PV: PyTorch policy/value + ONNX Runtime + dedicated PV search

Runnable first slice for **Ironclad vs Champ**, consuming the agreed **`combat_v4`** recordings.
No custom C++ neural-network runtime or weight format. PV has its own policy/value tree; it does not run the legacy UCB tree.

```
combat_v4 fights + search visits
          │ canonical native replay + shared feature encoder
          ▼
Python PyTorch learner ──► model.onnx
                              │
                   C++ ONNX Runtime evaluator
                              ↕ V(s), legal-action logits
                   dedicated public-belief PV tree
                              │ actions + visits + terminal outcome
                              └──► replay / retrain
```

## Pieces

- `model.py`: token embeddings, sum pooling, small shared trunk, value and legal-action heads; ONNX export.
- `features.cpp`: reuse existing combat encodings for both replayed training states and search leaves.
- `data.py`: join `combat_v4` fights/search Parquet, replay them, write Parquet feature shards.
- `train.py`: streaming bootstrap/retraining; PyTorch checkpoints and ONNX exports.
- `evaluator.cpp`: thin batched CPU ONNX Runtime call, with contract validation and one inference thread.
- `search.cpp`: dedicated public-belief PV tree with batched policy/value predictions and optional root noise.
- `apps/pv/worker.cpp`: replay/encode, inference parity interface, and play/record. Python owns orchestration.

## Build

```bash
.venv/bin/python -m pip install -r requirements-onnx.txt
apps/pv/setup_runtime.sh  # matching official headers + wheel library link, all under build/deps/
cmake -S . -B build/pv -DCMAKE_BUILD_TYPE=Release
cmake --build build/pv --target pv_worker pv_features_test encoding_v4_test -j6
ctest --test-dir build/pv -R 'pv_features_test|encoding_v4_test' --output-on-failure
```

Without prepared C++ support, ordinary builds omit PV rather than downloading dependencies automatically.
An alternative runtime installation can be selected using `-DPV_ORT_ROOT=...`.

## Bootstrap

```bash
WORKER=build/pv/agents/combat/pv/pv_worker
RUN=runs/pv/champ
.venv/bin/python -m agents.combat.pv.data --fights "$COMBAT_DIR"/fights-*.parquet \
  --search "$COMBAT_DIR"/search-*.parquet --encounters 39 --out "$RUN/rows"   # one Parquet shard, rows.parquet
.venv/bin/python -m agents.combat.pv.train --data "$RUN/rows/rows.parquet" --out "$RUN/iteration-0" --device cuda
```

Shard layout: `agents/combat/pv/data.py` docstring. Training logs per-epoch train/val value loss, policy loss and val
top-1 agreement with the most-visited move; `--stream` re-reads one shard at a time instead of holding all in RAM.
A replay mismatch or unsupported feature aborts without publishing a shard.
Value targets, network outputs, terminal evaluations and tree backups all use **100 × win probability units**:

```text
defeat  → 0
victory → 100
```

The terminal objective lives in `CombatObjective`; there is no leaf/root max-HP conversion, denominator or score clamp.
The value head is nonnegative and unbounded (`100 × softplus`). `VALUE_SCALE=100` is a fixed learning-conditioning
scale: value loss is `mean(((prediction - target) / 100)²)`, added to policy cross-entropy. It does not change search units.
`--value-mix M` uses `M·100·won + (1−M)·PV search root_value` (roots must be in [0,100]); no-root/forced rows
keep the outcome target. Default M=1 is outcome-only. Old HP-unit `--teacher-root-mix` remains unsupported (must be 1).
Non-finite or negative predictions/targets fail rather than being clipped or replaced.

Policy targets are normalized MCTS child visit counts. Missing/zero-visit search rows have no policy loss;
invalid visits or unmatched recorded actions abort. Played actions are not substituted for expert distributions.

Validation membership is SHA-256(run seed) modulo 10 = 0, so all fights/policy variants of a run stay together.
This is a deterministic approximately 10% split, not a guarantee of exactly 10% on a small dataset.
Both training and validation seeds must be present. Optional `--init previous/model.pt` starts a retraining round.

## Play / next iteration

`apps/pv/play.py` plays a parquet of starts (`fight_id`, `start`) with N `pv_worker play|teacher` processes and writes
combat_v4 `fights-0`/`search-0`, `decisions_stats-0` (seconds, simulations, tree depth in actions and player turns,
nodes; depth null for the teacher) and `summary.json` (see its docstring). PV plays on a BattleContext with its own
legal-action list, so recorded bits replay with `combat_v4::replay`.

```bash
.venv/bin/python apps/pv/play.py --starts STARTS.parquet --agent pv --model "$RUN/iteration-0/model.onnx" \
  --sims 1000 --explore --workers 10 --out "$RUN/selfplay"
.venv/bin/python apps/pv/play.py --starts STARTS.parquet --agent teacher --sims 20000 --workers 10 --out OUT
```

Omit `--explore` for deterministic evaluation. Native play uses eight public-belief particles and records root visits;
forced moves do not get invented policy labels. A 50-turn/512-action guard produces `status=capped`, which encoding skips,
never relabels as defeat. The JSONL play interface is an experimental transport, not a replacement replay schema.
No scheduling, promotion policy, or large collection/evaluation run is launched automatically.

## Search lifecycle and policies

```text
observed root → one network evaluation → construct Search with root prediction
                                               │
                  select/reserve paths → batch distinct unevaluated leaves
                                               │
                     evaluate → assign priors → back up values → repeat
```

A fresh tree is built for each non-forced decision by default. `--oracle` instead searches the true state with one
particle and keeps the chosen child subtree; SIMS is new simulations per decision. No transpositions are used.
`--policy-only` skips search and chooses argmax network logits. Oracle self-play uses `--explore --sample-turns`: root
noise plus moves sampled proportional to visits in player turns 1–2, argmax thereafter. Root evaluation is outside the simulation budget.
Construction requires valid root predictions and particles matching the root's public observation/legal menu.
There is no UCB/PUCT mode flag, no implicit root initialization and no uniform prior fallback.
The opt-in `--rollout-mix` leaf option mixes network values with guided-rollout terminal values (default off).
Pending nodes are never traversed: multiple reserved paths share one evaluation and each receives a backup.
Virtual loss treats in-flight paths as temporary zero-value visits. A 512-action path cutoff uses the reached node's V.

Defaults are AlphaZero-style PUCT scoring and argmax traversal:

```text
Q = value_sum / (visits + in_flight), or parent network value for an unvisited edge
U = c × prior × sqrt(1 + parent_visits + parent_in_flight) / (1 + visits + in_flight)
select argmax(minmax(Q) + U)
```

The parent pseudovisit makes priors guide the first traversal. `c=1.25` applies to tree-wide min-max normalized Q. There is no inherited first-play margin or uniform policy floor.
Traversal selection and the played move are distinct: play chooses the most-visited root edge, with stable ties.
`--explore` mixes 25% Dirichlet root noise (alpha 0.3). `--sample-turns` separately samples early-turn played moves.

`Search<CustomScore, CustomSelection>` changes these policies without runtime algorithm flags. Score receives
`(Q, prior, parent_visits, edge_visits)`; selection receives `(scores, RNG)` and returns an edge index.
Invalid scores or indices fail. Traversal remains argmax by default; played-move sampling is an independent self-play option.

PV reuses the legacy tree's static public-observation/action-key utilities and the existing particle sampler,
**not its search driver**. The exposed sampler ignores the legacy teacher's process-wide tweaks.
Rune Dome **search is rejected**: current particle sampling preserves current enemy intents and would expose hidden
information. Encoding masks those features, but correct hidden-intent belief sampling is still required before playing Dome.

## Input/value contract: `pv_champ_win_v3`

ONNX inputs are float32; categorical fields are integer-valued and converted to IDs in PyTorch:

| Input | Shape | Contents |
|---|---|---|
| context | B × 65 | existing global/player features, input state/task, scaled max HP |
| cards | B × C × 18 | existing card token; pooled separately by hand/draw/discard/exhaust |
| monsters | B × M × 29 | existing monster token plus Champ stance count and phase |
| potions | B × P × 19 | potion ID + existing 18 mechanics features |
| relics | B × R × 4 | existing relic ID + three counter/condition features |
| actions | B × A × 261 | kind/task/flags, source card, target monster, potion, interaction, selected subset, validity |

The v3 contract is win-only (100·won), retaining the softplus head and scaled MSE. Old normalized caches, PyTorch checkpoints and ONNX models are
incompatible and rejected; regenerate caches and retrain. Do not resume an old checkpoint under the new objective.

All token counts and batch size are dynamic. Zero-ID set rows are padding; actions carry a separate validity flag.
Subsets embed up to ten selected card tokens, not opaque hand-index masks. Outputs are `value[B]` and `policy_logits[B,A]`.
Draw order and RNG state are excluded. Opaque v4 monster counter/unique-power fields are cleared rather than exposing
raw simulator storage. Rune Dome hides intent-derived features and Champ counters that update while choosing an intent. The shared encoder has an **opt-in** event-card extension (Apparition, Bite, JAX, Ritual Dagger);
legacy callers keep their original capability boundary and frozen feature layouts.

## Verification status / remaining

- Before this documentation update: the redesigned tree built; native checks covered objective units, PUCT scoring,
  root priors, budget/backup accounting, pending-leaf coalescing, custom template policies and invalid predictions.
- One redesigned play/record/replay smoke check completed (one fight, 28 actions/replayed states, exactly 32 visits
  per searched root). This is plumbing evidence, not a strength estimate. Old ONNX contracts were rejected.
- Earlier parity and one-epoch bootstrap checks exercised the previous normalized-value implementation; they do
  **not** establish training correctness for the win-only contract. Updated training and final regression
  checks remain pending. The deleted ONNX test has not been restored. No tests were run for this docs update.
- No model has been promoted. Search depth, realistic throughput, held-out win rates, broader encounter coverage
  and best settings remain unmeasured. Feature caches are Parquet; JSON lines are pipe transport only.

Visual primer: [combat-search docs](../../../docs/research/combat-search/README.md).
