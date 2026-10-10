# Human-deck combat expert iteration

Entry point: `python -m apps.run_rl.combat_loop`. This is a separate combat controller
inside the existing run_rl app; it does not alter the overworld loop. It reuses
`apps.human_champ.bench` for failure-tolerant native play and `agents.combat.pv` for
canonical replay, feature encoding, PyTorch training and ONNX export. No new native
search, architecture, weight format, or gameplay augmentation.

## First real round

```sh
PYTHONPATH=. .venv/bin/python -m runs.run combat_v4 human-combat-r01 --no-compact \
  --input megacrit_runs_v1/2026-10-05/human-train-pull2020 \
  --input human_champ_bench_v1/2026-10-05/v1 \
  --input combat_v4/2026-10-05/champ-diag-D5 -- \
  .venv/bin/python -m apps.run_rl.combat_loop \
  --runs runs/schema=megacrit_runs_v1/date=2026-10-05/id=human-train-pull2020 \
  --regression runs/schema=human_champ_bench_v1/date=2026-10-05/id=v1/out/starts.parquet \
  --init runs/schema=combat_v4/date=2026-10-05/id=champ-diag-D5/model \
  --rounds 1 --max-train-decks 800 --teacher-decks 128 \
  --sims 2000 --teacher-sims 20000 --workers 8 --epochs 3 --device cuda --out '{out}'
```

Run the bounded pull first; do not train from a live-growing source for the real round.
The pipeline smoke used an explicitly partial source snapshot and tiny budgets, only
for plumbing; its model is not used to initialize the real run.

## What happens

1. Copy/freeze the native worker and initial model. Persist configuration/checksums.
2. Reconstruct exact-deck supported starts from the new pull, reusing the benchmark
   converter. Keep one human-seed start per source deck (no augmented training copies).
3. Exclude existing regression benchmark play IDs AND deck/relic families. Partition
   new families approximately 90/10; all same-family source decks stay together.
   The persisted `split.json` maps each training/validation fight ID explicitly;
   absent IDs fail closed. The PV learner's legacy seed-hash default remains unchanged
   for other workflows. This controller always supplies the deck-based manifest.
4. Freeze baseline validation play and a deterministic 100-deck regression subset
   (both regression seeds). Retain baseline validation trajectories solely for loss
   diagnostics; they never enter gradient updates.
5. Collect up to 800 training decks with D5 + public-belief action search, 2k sims,
   root noise and early-turn visit sampling. Canonically replay/encode completed plays.
6. Select up to 128 completed training starts for full MCTS20k trajectories: half
   deterministic coverage, then learner losses preferentially. This is **teacher
   recovery/coverage trajectories**, not teacher annotations of learner intermediate
   states. Teacher and learner versions receive distinct fight IDs, same split.
7. Train from learner + teacher states, outcome-only values, and concentration-weighted
   policy targets. Current defaults: 3 epochs, AdamW lr=1e-4, gradient norm cap=1.
8. Play the candidate on identical validation/regression starts at 2k sims; write paired
   comparisons and a selected-model pointer. Never overwrite the original D5 model.

Optional multiple rounds consume disjoint source-deck batches, not repeated fixed-seed
copies. Recent shards form a replay window. Rounds stop when the available source pool
is exhausted; collecting more decks is explicit, not an automatic infinite download.

## Flat policy weighting

Opt-in PV flag: `--flat-policy-weighting`.
For n legal actions and p_max from the unsharpened visit target:

`weight = clamp((n * p_max - 1) / (n - 1), 0, 1)` (single-action rows get zero).

Uniform targets receive no policy gradient, near-flat ones little, one-hot ones full.
Legal actions include zero-visit choices; padded slots do not count. Value labels and
value loss are unchanged, including losing states. Policy loss remains divided by the
original policy-state count, not by weight sum: flat rows genuinely contribute less.
Training logs include effective weighted policy-state mass. This is a concentration
heuristic, **not** calibrated teacher confidence; its benefit must be measured by play.
Existing training defaults and checkpoint/ONNX contracts remain compatible.

## Evaluation and promotion

Report deck-mean rates, paired difference, and 1 SE clustered by deck/relic family.
A deck contributes only if all its evaluation starts completed in both arms. Errors and
caps are reported separately, never labeled as losses. Require >=95% paired coverage,
at least 20 validation decks, positive validation difference >1.96 SE, and no regression
difference below -1.96 SE. This is a conservative practical gate, not a proof of no
regression or a multiple-comparison-adjusted guarantee. If uncertain, keep incumbent.

Validation outcome labels reflect frozen baseline continuation, not optimal play.
Actual paired gameplay is the promotion metric. The old human benchmark remains a
regression set, not an untouched final test (we have already inspected it). Historical
balance, absent potions, canonical deck order and reset-counter caveats still apply.

## Progress, failures, resumption

Managed logs: `<run>/logs/stdout.log`, `stderr.log`.
Stage logs: `<run>/out/logs/*.log` and `out/iter000/logs/*.log`.
Results: `out/REPORT.md`, `summary.json`, per-round `comparison.json`, model and replay
shards. Per-fight journals flush immediately; completed combat_v4 fights and search
annotations are exported together. Worker/model/start checksums protect play resumption.

The controller uses an exclusive output-directory lock. Stage success markers skip
completed work; direct rerun with **identical** settings/output resumes an interrupted
controller. runs.run needs a fresh ID for a new run; do not use --overwrite to resume.
It waits for other PV gameplay rather than stopping it, and uses at most 8 workers.
Encoder/training failures stop the pipeline with their logs; they are not concealed.

Tests: `python -m unittest agents.combat.pv.test_human_training apps.run_rl.test_combat_loop
apps.human_champ.test_bench agents.combat.pv.test_train` (also CTest-discovered).
