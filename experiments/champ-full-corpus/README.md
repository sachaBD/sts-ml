# Champ full corpus

## Objective

Cold-start an autonomous Ironclad A20 Champ agent on the usable Mega Crit corpus, including human wins and losses.
Improve unseen-composition performance without losing legacy-50 performance. One training run: reported evaluation
uncertainty does not measure training-run variability.

## Frozen inputs — 2026-11-02

`runs/benchmark/champ-full-corpus-v1/frozen/` contains the inputs, `selection.json`, `summary.json` and `sha256.json`.
The parent directory contains the launch config snapshot and `preflight.json`. No training has been launched.

| Partition | Decks | Share of 9,941 retained loadouts | Fixed fights/deck | Evaluation fights |
|---|---:|---:|---:|---:|
| Training | 7,941 | 79.88% | sampled | — |
| Legacy validation | 50 | 0.50% | 10 | 500 |
| New validation | 950 | 9.56% | 2 | 1,900 |
| Final test | 1,000 | 10.06% | 10 | 10,000 |

The candidate pool has 9,942 loadouts and 9,936 upgrade-agnostic card compositions (multiplicity preserved).
All 50 legacy compositions occur exactly once. Their corpus equivalents are replaced by the original legacy
loadouts. Composition groups never cross partitions. Five duplicate-composition variants remain in training;
one extra variant of a held-out composition is explicitly withheld, giving 9,941 retained loadouts.

- **Legacy starts are unchanged records** from `runs/benchmark/champ-50-inputs/frozen/validation.jsonl`:
  ten per deck selected deterministically, with identical fight IDs, seeds, HP, potions and relic state.
  **No jitter, reseeding or potion resampling.** Historical comparisons must use these exact matching fight IDs.
- New validation and final: deterministic composition selection, human HP and fixed seeds/potions. No selection by
  human or agent wins. Realized starts are frozen rather than redrawn on each evaluation.
- New validation is primary; legacy retention is a separate secondary evaluation file/channel (the app calls this
  `in_sample_fights`, but these legacy compositions are excluded from this experiment's training).
- Earlier converter-generated 96/2/2 partitions are superseded. Do not use those validation/final files.

## What will run

`train.toml` starts a new entity model (192 dimensions, four layers, six heads), seed 0, with no initial checkpoint.
Each update collects 10,000 fights: uniform deck draws with replacement, HP jitter 0.1 and sampled potions. Search:
16 simulations, eight particles, Gumbel scale 1; rollout mix 0.5 falls to zero at update 30.

Learning: 500 AdamW steps of 2,048 sampled rows, learning rate 0.001, uniform replay over ten updates (100k fights),
no row cap. **Launch still requires explicit permission.** Disk cleanup is complete. The buffer is disk-indexed, with only
sampled minibatches read and length-bounded padding. Expired replay is pruned. Worker: nine actors × 224 slots; GPU batches 512/256/32, 64-token GPU capacity with 24
attended slots and mixed precision; larger/overflowing states use the full CPU model.

Evaluate initially and every three updates, 64 simulations, no Gumbel noise or rollout blending: 1,900 primary
and 500 unchanged legacy fights. Final fights are copied with the run but never passed to routine evaluation or
replay. Runs until explicitly stopped; minimum disk reserve 10 GiB, at most two recognised-CUDA recovery retries.

From the main checkout (config snapshot already exists):

```sh
./apps/gumbel_nn_combat_expert_iteration/run.sh runs/benchmark/champ-full-corpus-v1/train.toml
```

That command launches training. It has **not** been executed. Standard stop/status commands apply.

## Replay ratio and capacity planning

Training replay ratio R = optimizer row draws / newly collected decision rows. Each update draws
500 × 2,048 = 1,024,000 training rows. Scoring rows are not optimizer draws.

- Previous champ100 updates 14–18: 10k fights produced 301,405–308,055 rows and 2.865–2.931 GiB per update;
  R was 3.32–3.40. Five consecutive updates from one run, without sampled potions, at 128 search sims.
- The 448-fight potion probe at 16 sims averaged 38.42 decisions/fight. Scaling that workload to 10k fights gives
  R ≈ 2.66 and 3.81 GiB/update. One small probe; not a prediction interval for this full-corpus run.
- Changing buffer size does not change R at a fixed collection/training budget. It spreads draws over more history.
  The agreed ten-update window retains up to 100k fights.
- Pruning happens after collection/training. Ten retained updates therefore need room for eleven sample batches
  temporarily, plus the 10 GiB reserve: about 42 GiB using the previous run's workload, or 52 GiB scaling the potion
  probe, before checkpoint growth. Approved cleanup reclaimed 23.36 GiB, leaving 59.16 GiB free: older ONNX exports
  and stale uv build environments removed; newest two checkpoints' exports per run and every checkpoint preserved.
  Journal: `scratch/champ-full-corpus/cleanup-journal.json`. `max_train_rows` caps the replay index, not sample bytes.

## Evaluation and checkpoint selection

Report legacy retention and new-deck generalisation separately. Primary: macro win rate over the 950 new validation
decks. Two fights per new deck are sparse; individual-deck estimates are noisy, breadth is the goal. Report paired
changes on fixed starts with uncertainty clustered by composition; intervals are conditional on this training run
and not adjusted for repeated validation looks.

Choose the checkpoint with the best committed primary score, ties to the earlier checkpoint. Freeze its identity
before final evaluation. Compare it and checkpoint 0 on the same reserved final fights; never use final results to
select another checkpoint.

Historical scores have simulator-version confounding. Identical legacy starts enable matched comparisons, not a
claim that all historical policies used identical simulator rules or search settings.

## Qualification and limitations

- Clean Release worker: `build/champ-full-corpus/agents/combat/nn/nn_play`, from committed native source
  `7687ce4943c8cc56f1a38e931954ba088e051d4f`, lightspeed `bdc3739b605f78d52914180b868578cb887b7754`.
  Binary/config hashes are in `preflight.json`; unrelated local native edits were not built.
- All 20,341 frozen starts passed native import/encoding. Final starts were checked structurally, **not played**.
- Partition invariants, unchanged legacy records and hashes verified. Native encoding/decision/explanation tests
  and 28 Python environment/app tests passed. A disposable cold CUDA model produced a valid output on one
  16-row probe minibatch; no optimizer step or weights saved. No export/startup or full training iteration rehearsed.
- Logs: `scratch/champ-full-corpus/`. 59.16 GiB free after cleanup, not a guarantee for an unlimited run.
- Dataset reconstruction is approximate: unknown relic counters/bottled assignments; 353 converted decks with unknown
  card misc; 314 converted decks with multi-upgrade Searing Blow collapsed to boolean upgraded. These counts describe
  the pre-capability-filter conversion, not necessarily the final training partition.
- Native capability checks cover initial states, not all future effects. Dome and Shard were excluded.
- Unsplit source: `runs/benchmark/megacrit-ironclad-a20-candidates-20261102/decks.jsonl`.
  Detailed conversion audit:
  `runs/schema=pv_starts_v1/date=2026-11-02/id=ironclad-a20-all-20261102-v2/out/dataset-audit.json`.
