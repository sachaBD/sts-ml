# Card-outcome overnight run: morning report (Opus, 2026-09-30 07:05 BST)

## Summary
- **Outcome prediction:** the augmented model (natural plus synthetic data) is clearly better than the natural-only model on boss and elite fights. More data helped from s1 to s3, then stopped helping at s3c. Across 3 training seeds, supplementary-dev boss Brier went .207 → .182 → .172 → .180, against .243 for natural-only. On easy and hard fights both models are near perfect, so neither category tells the models apart.
- **Card effect (take vs skip):** this is the main question. Only the augmented model learns anything beyond predicting zero, and the gain grows with data. The clearest case is elites: dMSE vs zero +0.45 → −1.38 → −1.93 → −2.06. That is below zero for every seed from s2 on, and roughly 10% of the true between-card variance. Bosses are weak or borderline (−0.35, with seeds ranging −0.79 to +0.55). Easy fights show a tiny but consistent gain (−0.12); hard fights show roughly nothing. The natural-only model never beats predicting zero.
- **The metric the runbook asked for is unreliable here.** With 3–4 seeds, each measured per-pair card effect is mostly noise. Split-half correlation between seeds is 0.01–0.25, so MAE against "zero effect" hardly moves for any model. I therefore also report an unbiased score, dMSE = mean(pred² − 2·pred·obs). Observation noise cancels out of it, and negative means better than predicting zero.
- **Recommendation:** returns from more data under this small fixed topology flattened at s3c. The next experiment should be a modest capacity/topology sweep on the cumulative ~330k fights, judged on card-effect dMSE with several seeds. Along with it, build a dedicated high-seed card-effect evaluation set (≥16 seeds, bosses and elites) so that effect error can be measured directly.

## Checkpoints: data and compute
Fights generated, with generation minutes in brackets (11 workers). Death counts are in DECISIONS.md.

| Checkpoint | Easy | Hard/elite | Boss | Cumulative fights | Gen time |
|---|---|---|---|---|---|
| s1 (also fixed eval cohort) | 32,000 (1.9) | 12,000 (11.2) | 1,980 (18.2) | 46k | 31 min |
| s2 | +40,000 (2.3) | +30,000 (24.8) | +4,500 (37.8) | 135k* | 95 min |
| s3 | +60,000 (3.6) | +84,000 (70.2) | +16,500 (143.3) | 295k | 217 min |
| s3c (extension) | – | +30,000 (26.7) | +5,400 (47.0) | 331k | 74 min |

\*s2 also includes a dev-only supplementary cohort ("devsupp", natural donor bucket 8, 4 seeds): 12,000 hard/elite and 2,400 boss fights. None of it is used for training.

From s2 on, training data uses 2 seeds per group (more distinct decks rather than repeated seeds) and donor buckets 4–7 only. Each training run took 1–4 minutes on the GPU. All losses were finite. The augmented arm early-stopped at epoch 3–7 in every run (natural-only arm: 7–14). That early stopping suggests the small topology fits quickly.

## Outcome and card-effect metrics
**Main table:** supplementary dev (fixed eval plus devsupp: 41 boss and 60 elite donor clusters). Augmented model, mean (min..max) over 3 training seeds. Full table: `diag/seedtable.md`.

| Metric | Natural-only | s1 | s2 | s3 | s3c |
|---|---|---|---|---|---|
| Brier, natural boss | .208 | .170 (.164–.180) | .166 | .155 (.150–.162) | .162 |
| Brier, synthetic boss | .243 | .207 | .182 | .172 | .180 (.171–.187) |
| Brier, natural elite | .052 | .040 | .039 | .038 | .038 |
| Brier, synthetic elite | .053 | .054 | .046 | .046 | .048 |
| Card dMSE vs zero, elite | +0.61 | +0.45 (−1.07..+1.25) | −1.38 (−2.32..−0.68) | −1.93 (−2.50..−1.61) | −2.06 (−2.31..−1.91) |
| Card dMSE vs zero, boss | +0.54 | +1.19 (−0.90..+2.46) | +0.06 | −0.59 (−1.06..+0.22) | −0.35 (−0.79..+0.55) |
| Card dMSE vs zero, easy | +0.05 | +0.14 | −0.09 | −0.13 | −0.12 |
| Card dMSE vs zero, hard | +0.28 | +0.57 | −0.09 | +0.08 | −0.08 |

**Fixed eval cohort only:** the runbook's fixed comparison, training seed 0 only. There are only 12–14 boss/elite donor clusters, so these numbers are noisy.

| Metric (augmented) | s1 | s2 | s3 | s3c |
|---|---|---|---|---|
| Brier, natural boss (natural-only .210) | .164 | .168 | .162 | .168 |
| Brier, synthetic boss (natural-only .267) | .176 | .189 | .186 | .205 |
| Card dMSE, elite (±cluster SE) | −1.39±0.92 | −2.80±1.26 | −1.25±0.97 | −2.29±0.88 |
| Card dMSE, boss | +0.56±1.87 | +1.57±1.42 | +2.05±2.37 | +0.28±1.68 |

The per-encounter card-effect MAE vs zero is in each run's `out/report.md`. Across all models, it stays within a few percent of the zero-effect baseline.

## Caveats
- These are development-set results. They have been inspected repeatedly and should be read as exploratory, not as confirmation. They say nothing about gameplay strength.
- **Training variance matters.** One seed alone was misleading: s2 with seed 0 looked like the best card-effect model. Three seeds is still a small sample.
- **Noisy fixed cohort.** The fixed eval cohort has few boss/elite clusters, so boss conclusions rest on the supplementary cohort, which I built and labelled.
- **Label policy.** Natural labels come from an older policy. The realistic boss states generated here do not settle the earlier mismatch with the low-resource gauntlet.
- **Code changes:** two opt-in environment overrides in `apps/combat_transition/train_marginals.py`, `CARD_OUTCOME_LR` and `CARD_OUTCOME_SEED`. Defaults are unchanged; runs with default settings reproduced the earlier results exactly.
- **lr diagnostic:** a run with learning rate .001 gave no gain.
- **Split integrity:** generator configs used new seeds and donor buckets 4–7 for training. The deduplication check passed on every training run.
- **Failed launch:** one stage-3c launch failed on a relative-path argument before any work started. That log is kept as `logs/stage3c-gen.failed-path.log`.

## Artifacts
- `DECISIONS.md` records the rationale at each checkpoint. Configs are in `configs/`, run manifests in `data/`, and generation logs in `logs/`.
- Model runs: `runs/schema=combat_outcome_v1/date=2026-09-3*/id=card-outcomes-{s1,s2,s3,s3c}-model[-supp][-seedN]`.
- Data runs: `runs/schema=card_marginals_v1/date=2026-09-29/id=card-outcomes-*`.
- Diagnostics: `diag/reliability.py` (noise and unbiased dMSE), `diag/compare.py`, `diag/seedtable.py`.
