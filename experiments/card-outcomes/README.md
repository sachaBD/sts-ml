# Staged card-outcome experiments

Follow **RUNBOOK.md**. There is no unattended seven-hour loop: Opus runs one checkpoint, reviews it, then chooses the next.

## Stage 1 (~45 minutes generation, then train/evaluate)

The supplied configs cover all categories with 11 workers: easy 200 groups × 4 seeds, hard/elite 100 × 4, boss 33 × 3. These are starting allocations, not a runtime guarantee; adjust to user-owned preflight measurements before launch. Smaller repetition counts favour initial deck diversity; first card-effect estimates will be noisy.

From the repo root, after generator preflight:

```bash
D=experiments/card-outcomes
mkdir -p "$D/data" "$D/logs"
"$D/generate.sh" "$D/data/s1.txt" "$D/configs/s1-easy.toml" \
  "$D/configs/s1-hard-elite.toml" "$D/configs/s1-boss.toml" &&
cp -n "$D/data/s1.txt" "$D/data/eval.txt" &&
"$D/train.sh" card-outcomes-s1-model "$D/data/s1.txt" "$D/data/eval.txt"
```

Run the command in the background with an outer log and arrange the Opus completion notification using the supervising harness. The scripts do not start/invoke agents. Opus must set an absolute nine-hour finish time and bound each stage to leave room for training/reporting. Generation is finite; any failed job stops the shell sequence. No jobs are launched by these instructions alone.

## Later checkpoints (~2 hours, then 2–4 hours generation)

Opus copies/edits finite configs with **new run IDs and generator seeds**, chooses category allocation, then:

```bash
D=experiments/card-outcomes
"$D/generate.sh" "$D/data/s2.txt" CONFIG1.toml CONFIG2.toml CONFIG3.toml &&
cat "$D/data/s1.txt" "$D/data/s2.txt" > "$D/data/through-s2.txt" &&
"$D/train.sh" card-outcomes-s2-model "$D/data/through-s2.txt" "$D/data/eval.txt"
```

Repeat with cumulative data for stage 3. **Never change `eval.txt`:** it supplies the same synthetic development AND early-stop rows at every checkpoint. New generated runs contribute training rows only. Natural dev/early-stop states stay fixed too. Do not repeat earlier generator seeds for the same stage; duplicate observations are rejected.

## Existing apps, not another execution framework

- `generate.sh` calls `apps/card_marginals/run.sh` for each config; that app builds/snapshots its worker and uses the standard run launcher. It writes completed run IDs to the requested manifest.
- `train.sh` invokes `apps/combat_transition/train_marginals.py` via `sts_combat_rl.run`. It trains/evaluates natural-only and augmented models on GPU, fixed topology. Optional fourth argument sets maximum epochs (default 60).
- Run artifacts/logs remain under `runs/`. Generation wrapper logs are `experiments/card-outcomes/logs/<run-id>.log`.
- Training output: `natural.pt`, `augmented.pt`, `report.md`, `report.json`, per-fight dev predictions. Reports show category/encounter outcome metrics and paired card-effect MAE versus zero effect.
- Do not modify/rebuild the teacher between checkpoints without recording the policy change. Each generator run records a worker hash; the app snapshots its executable. Historical natural labels are an older-policy reference. Source/build provenance still matters in this shared dirty working tree.

## Labels, splitting and weighting

No new outcome labels are needed. The adapter uses `won`, `battle_final_hp` (before Burning Blood), `pre`, encounter/category, variant, shared seed, group and donor lineage. It supports legacy easy-pilot rows too; the supplied plan generates fresh easy data with the preflighted teacher.

- Hard/elite/boss: donor buckets 4–7 train, 8 dev, 9 early stop. Natural bucket 8 is also excluded from training; natural dev uses 0–1 plus 8. Natural 2–3 remain reserved.
- Easy synthetic: generated-group buckets 4–8 train, 0–1 dev, 9 early stop; 2–3 excluded. Alternatives/repetitions stay together.
- Category-balanced training; equal natural/synthetic weight within a category when both exist. Events excluded. Maximum 32,768 sampled examples per epoch, patience 8. Model topology unchanged.
- Generator aggregate HP/counter distributions use the source cohort; these are exploratory development comparisons, not pristine confirmation. Fixed realistic-boss tests do not certify the early-gauntlet resource profile.

## Checks performed

Grouping/dedup, paired-score handling, real legacy parquet encoding, tiny GPU fit and a two-arm train/evaluate/checkpoint/report smoke. No full research run has been launched. Generator preflight remains user-owned.

```bash
PYTHONPATH=python:. .venv/bin/python tests/test_card_outcome_training.py
```
