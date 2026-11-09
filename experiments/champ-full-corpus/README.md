# Champ full corpus

## Frozen partition preparation

`prepare.py` groups decks by card-ID multiplicity, ignoring upgrades. It reserves 950 unseen
compositions for validation and 1,000 for final testing; all 50 legacy compositions are excluded
from training. Equivalent held-out variants are withheld. Selection does not use fight outcomes.
Legacy evaluation retains ten unchanged historical starts per deck; new validation/final starts
use deterministic seeds and sampled potions.

```sh
PYTHONPATH=. .venv/bin/python experiments/champ-full-corpus/prepare.py \
  --pool runs/benchmark/megacrit-ironclad-a20-candidates-20261102/decks.jsonl \
  --legacy runs/benchmark/champ-50-inputs/frozen \
  --out runs/benchmark/champ-full-corpus-v1/frozen
```

The output directory must not exist. The original v1 partition contains 7,941 training loadouts,
950 validation decks, 50 legacy decks and 1,000 final decks. `selection.json`, `summary.json` and
`sha256.json` record membership, counts and input/output hashes. Final starts are reserved, not
routine evaluation inputs. These are the v1 inputs; v2 uses a separately frozen corpus.

## Legacy evaluation sets

`legacy_core` uses 50 historical fights for each of the 10 deeply studied decks.
`legacy_extended` retains 10 fights for each of the other 40 decks. Generate from the repo root:

```sh
PYTHONPATH=. .venv/bin/python experiments/champ-full-corpus/split_legacy.py \
  runs/benchmark/champ-full-corpus-v1/frozen/legacy.jsonl \
  runs/benchmark/champ-50-inputs/frozen/validation.jsonl \
  runs/benchmark/legacy-core50-extended10
```

The v1 and v2 frozen legacy files are byte-identical, so they share these inputs.

## Inventory48 cold run — 2026-11-05

Registered and approved for launch as
`gumbel_nn_combat_v1/2026-11-05/champ-full-corpus-v2-inventory48-outcome-aux-cold`.
Config: [`train-v2-inventory48-64s-cold.toml`](train-v2-inventory48-64s-cold.toml).

- New `inventory_v1` network, width 192, four layers, six heads; outcome and all auxiliary heads retained.
- Fresh weights/replay; 48-token primary, 128-token wide fallback, full CPU fallback without truncation.
- Prior recipe otherwise unchanged: 10,000 games/update, 64 simulations, eight particles, nine actors × 224 slots;
  500 training steps, policy batch 2048, combat-balanced value batch 128, replay window 10.
- LR 0.0005, half-life 30, floor 0.0001; rollout mix remains 0. No bootstrap change.
- Same frozen v2 training/validation and legacy evaluation sets; evaluation at initialization and every three updates.
  Reserved final fights remain unused for selection.
- Launch from the pinned `combat_rl-inventory` worktree using its release `build/nightly` worker.
  The standard launcher records the source revision, command and input lineage in `run.json`, and freezes config/inputs.
- New dashboard: port **8002**. Retained previous run/dashboard: port **8001**.
- Preserve the entire previous warm-run `out/training/iter070`, including its checkpoint, replay and exports.
  Cleanup audit: `runs/analysis/inventory-nightly-cleanup-20261105/` in the primary checkout.

## Wide-batch-128 restart — 2026-11-05

User approved a fresh start (not resume) as
`gumbel_nn_combat_v1/2026-11-05/champ-full-corpus-v2-inventory48-wide128-outcome-aux-cold`.
Config: [`train-v2-inventory48-wide128-64s-cold.toml`](train-v2-inventory48-wide128-64s-cold.toml).
Only deployment change: wide GPU batch **32 → 128**; narrow remains two lanes with batch512/bank256,32,
wide remains one lane with 128-token capacity. All learning/search settings unchanged.
Fresh weights, optimizer and replay; prior inventory run is stopped, not overwritten.
Dashboard **8001** now follows this restart (supersedes the earlier 8002 registration).
The previous warm iteration70 remains protected in full.
