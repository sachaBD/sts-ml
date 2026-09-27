# Value net v2: work plan (GPU training, bigger head, wider encoders, auxiliary heads)

Scope, from `assessment.md` §7: step 1 (GPU training), step 3 (head), step 4 (wider cached encoders)
and step 5 (auxiliary heads). Step 2 (label consistency) and attention are deferred. Hold the label
setting fixed across all comparisons so the effect of the labels isn't mixed up with the effect of
the architecture.

**Label fix, assigned to the `fix-labels` agent:** a new opt-in `label = "shift"`.

- Decision rows are unchanged from `blend`.
- Child rows get v_c + blend·(z_parent − v_parent), where the parent is the decision row with the same
  (episode_id, decision_index). δ = 0 at or before the fight's random move.
- This keeps the teacher's ordering between siblings and removes the ~+0.065 chosen-action bonus on
  bosses.
- Details: `labels.md`, once written.
- Run the experiment ladder with a single label mode (preferably `shift` once it lands), plus one B0 run
  with `blend` to measure the label effect by itself.

## Guiding rules

- **Backward compatible.** v1 checkpoints and `value_weights.bin` files must keep loading and
  evaluating bit-identically: `tests/search_perf_test.cpp` compares against `tests/original_value_net.hpp`.
  v2 is chosen by `architecture.kind = "deep_sets_v2"` in the checkpoint config. A missing `kind` means
  v1.
- **Change one thing at a time** in the experiments. Each change is a config flag; the code can land
  together.
- **Judge by gameplay at equal wall-clock time**, not just validation MSE. Use `apps/value_play` on a
  fixed held-out fight set (a `query` over run seeds that no checkpoint trains on), then
  `apps/compare_fights`.
- Build dirs go under `build/<name>/` only (AGENTS.md).

## Workstreams and file ownership

Three agents work in parallel. Each owns the files listed and coordinates through the contract below.

### T: training infrastructure and auxiliary losses (steps 1 + 5, trainer side)

Owns: `python/sts_combat_rl/training/train_value.py`, `training/data.py`, `apps/value_train/train.py`
and new TOML configs.

1. **Device.** Add `device = "cuda" | "cpu"` (default "cpu", which keeps the old behaviour and
   determinism). Move batches to the GPU with non-blocking copies. `index_add_` pooling already works
   on the GPU.
2. **Throughput.**
   - Batch size 1024–4096.
   - `RowLoader` collate is numpy and single-process. Measure first; if it's the bottleneck, use pinned
     memory and a background prefetch thread or `num_workers`.
   - Target: well under 10 min per epoch over 4.5M rows.
3. **Memory** (15 GB RAM, ~8 GB free).
   - Loading all rows as float32 is ~5–6 GB. Store `cards.numeric` and the other numeric token arrays
     as float16 and convert to float32 per batch.
   - Optionally cache the packed `Rows` to a `.npz`/memmap next to the dataset so repeated runs don't
     go back through DuckDB.
4. **Optimiser.** Linear warmup (~1–2% of steps), then cosine decay. Keep AdamW.
   - Optional gradient clipping (norm 1.0).
   - EMA of the weights (decay ~0.999, config `ema`). Evaluate and save the EMA weights when enabled.
   - bf16/AMP is optional. The model is small, so fp32 is probably fine; measure.
5. **Reporting.** Per-category validation MSE already exists per encounter. Add a per-category
   (boss/elite/hard/easy/event) summary line. Add optional `category_weights` for the loss.
6. **Auxiliary losses** (consume the model's aux outputs, see the contract).
   - `won`: binary cross-entropy on `won_logit`.
   - `final_hp`: MSE on `hp` against `final_hp / starting_hp`. Check the columns and the
     `terminal_value` definition to pick the right normalisation. Target only wins, or all rows with
     0 on a loss; document the choice.
   - **Mask**: aux targets are trajectory outcomes. Only use rows where `row_kind = 'decision'` and
     `decision_index > last random move` (the same rule `assign_targets` uses for blending). Child rows
     and pre-random rows get weight 0.
   - Config: `aux_won_weight`, `aux_hp_weight`. Default 0 = off.
7. **Checkpoint.** Record the device, EMA, aux settings and the full `architecture` dict (including
   `kind`). The exporter writes whatever `architecture` contains.

Acceptance:
- A v1-architecture run on the CPU with the old config reproduces the old results (same code path).
- A v1-architecture run on the GPU over the new dataset finishes and gives similar or better validation
  MSE. **This is the new baseline (B0).**

### M: model v2 in PyTorch (steps 3 + 4)

Owns: `python/sts_combat_rl/models/deep_sets.py` (a new class `DeepSetsValueV2` or config switches) and
`models/__init__.py`. Also construction by `kind` for the trainer: provide a `build_model(architecture)`
helper that the trainer calls.

Architecture (every item is a config key; the defaults below are the proposed v2):

| Key | v1 | v2 default | Notes |
|---|---|---|---|
| `kind` | (absent) | `deep_sets_v2` | |
| `width` (token width) | 64 | 128 | Step 4: cached in C++, so cheap |
| `card_id_dim` | 8 | 32 | Step 4 |
| `monster_id_dim`, `move_dim` | 8 | 16 | Step 4 |
| `pool_count_features` | no | yes | Append log1p(count) for each of the 6 pools |
| `head_width` | 64 | 256 | Step 3 |
| `head_blocks` | 0 | 2 | Pre-LayerNorm residual blocks |
| `head_input_norm` | no | LayerNorm | Step 3 |
| `output` | tanh | sigmoid | Targets lie in [0, 1] |
| `aux_heads` | no | yes | Step 5: Python only |

Head, precisely:

```
f = concat(global_numeric, emb(input_state), emb(card_selection_task), 6 pools [, log1p(6 counts)])
f = LayerNorm(f)                                  # if head_input_norm
h = ReLU(Linear(len(f) -> head_width)(f))
repeat head_blocks:  h = h + Linear(W->W)(ReLU(Linear(W->W)(LayerNorm(h))))
h = LayerNorm(h)                                  # final norm when head_blocks > 0
value     = sigmoid(Linear(W->1)(h))              # or tanh (v1)
won_logit = Linear(W->1)(h)                       # aux_heads only
hp        = sigmoid(Linear(W->1)(h))              # aux_heads only
```

- Token encoders stay context-free 2-layer ReLU MLPs, so the C++ caches remain valid. Only the widths
  change.
- `forward` returns the value tensor as today, so existing callers are unchanged. Add
  `forward_all(...) -> dict(value, won_logit, hp)` or a `return_aux=True` flag for the trainer.
- Count features: count tokens per pool with `index_add_` of ones (card zones 0–3, monsters,
  interactions).
- Use stable parameter names, because C++ loads by name. Document them in the class docstring:
  `head_in_norm`, `head_in`, `head_blocks.{i}.norm/fc1/fc2`, `head_out_norm`, `value_out`,
  `won_out`, `hp_out`.
- Provide `@torch.no_grad()` golden outputs for parity testing: a small script or test that writes a
  handful of encoded states and their v2 values to JSON for the C++ agent.

Acceptance:
- A v1 checkpoint loads and gives identical outputs.
- v2 trains on the GPU (with T).
- The parameter count is reported. Expect ~300k–700k with the defaults.

### C: C++ inference for v2 (steps 3 + 4 on the search side)

Owns: `models/value_net.{hpp,cpp}`, `python/sts_combat_rl/training/export_value_weights.py` (only if
format changes are needed; the flat tensor-by-name format should already be enough), and new or
extended tests in `tests/`.

1. Dispatch on `config.architecture.kind`: v1 keeps the exact current code path. **v1 must remain bit
   identical** (search_perf_test).
2. v2: configurable embedding dims and token width.
   - `apply_fixed<64>` becomes a width-generic fast path. At least 128 and 256 should be fast.
   - LayerNorm (eps 1e-5, PyTorch semantics).
   - Pool count features.
   - Residual head blocks and sigmoid output.
   - Ignore `won_out` / `hp_out` (they may be present in the file).
3. Cache keys are unchanged: token encoders are still context-free.
4. **Parity test**: M's golden JSON goes through the C++ `ValueNet`, max abs diff ≤ 1e-5.
5. **Latency**: extend `apps/bench_search` or add a micro-benchmark. Report µs per evaluation for v1
   against v2 at default and smaller sizes (head 128, blocks 1), and simulations per second in a real
   search. This number decides the size at equal wall-clock time.

Acceptance: parity passes, v1 is bit-identical, and a latency table is written to
`slop_docs/topology/results.md`.

## Contract between agents

- **Checkpoint `architecture` dict**: `kind` plus all the table keys above. The trainer passes it to
  `build_model`, the exporter copies it into the `.bin` config, and C++ reads it.
- **Parameter names**: as listed under M. Any rename must be told to C before it lands.
- **Aux outputs**: `forward_all` returns `{"value": [B], "won_logit": [B], "hp": [B]}`, and the aux
  keys exist only if `aux_heads`.
- **Data**: default training query for all experiments (adjust if you hold out a gameplay set):
  `select * from combat_v3 where id = 'act1-all-bosses-a20-scaled-search'`.
  - Keep `label` and `blend` identical across runs.
  - Hold out a fixed set of run seeds for gameplay (e.g. `run_seed % 50 = 0`) and exclude it from
    every training query.

## Experiment ladder (after the code lands)

| Run | Change | Question |
|---|---|---|
| B0 | v1 arch, GPU trainer, new data | New baseline |
| E3 | + head (LN, log-count, 256×2 residual, sigmoid), width 64 | Does the head bottleneck matter? |
| E4 | E3 + token width 128, larger embeddings | Do cached encoders benefit from capacity? |
| E5 | E4 + aux heads (`won`, `hp`) | Data efficiency and generalisation |
| E5-small | E5 with head 128×1 | Accuracy vs latency trade-off |

For each run, record:
- validation MSE overall and per category;
- the train-vs-validation gap curve;
- µs per evaluation;
- `value_play` results on the held-out fights, using the same time budget or the simulations scaled by
  latency.

Write results to `slop_docs/topology/results.md`.
