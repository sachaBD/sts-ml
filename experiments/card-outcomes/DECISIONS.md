# Opus decision log (card-outcomes staged run)

Absolute finish: 2026-09-30 07:55 BST (start 22:57 BST, 9h).

## Stage 1 launch (22:58)
- Cost estimates from pilots: easy ~16k fights/2 min; hard/elite 12k fights ~11-20 min (pilot used tree_reuse=true, s1 false -> slower); boss ~6 s/fight -> 1485 fights ~15 min.
- Changes vs supplied configs: easy 200->400 groups, boss 33->44 groups (est. total ~40-45 min). Rationale: eval.txt is fixed from s1, so a larger s1 gives bigger fixed dev/early-stop cohorts, esp. easy (cheap) and boss (few donor buckets 8/9).
- Order: easy, hard/elite, boss. Log: experiments/card-outcomes/logs/stage1.log

## Stage 1 review (23:45)
- Data: easy 32k fights/1.9 min (39 deaths), hard/elite 12k/11.2 min (468 deaths), boss 1980/18.2 min (887 deaths). Training ~1 min, finite; best epoch ~5-7 (early overfit, small data).
- Outcomes: augmented beats natural-only on natural dev Brier for boss (.210->.164), elite (.054->.039), hard; HP MAE better everywhere. Synthetic elite Brier worse (.040->.052).
- Card effect (report MAE): neither model beats zero-effect MAE. Diagnostic diag/reliability.py: observed per-pair effects at 3-4 seeds are mostly noise (seed split-half r: easy .09, hard .03, elite .25, boss .01). MAE vs zero is thus a weak metric. Added unbiased dMSE vs zero (= mean(pred^2 - 2 pred*obs), noise cancels; cluster SE): augmented elite -1.39±0.92, hard -0.15±0.42, boss +0.56±1.87, easy ~0. Natural-only ~0 everywhere (predicts ~no effect).
- Fixed dev has only 12-14 donor clusters for boss/elite -> too small to resolve card effects.
## Stage 2 plan (~2h generation, launched ~23:47)
- Training data: distinct decks over repeated seeds (seeds 2) since per-fight loss does not use pairing; natural_buckets [4-7] only for hard/elite/boss (bucket 8/9 rows from non-eval runs are dropped anyway). Emphasis on boss/elite where outcome & card effects are largest and worst predicted: boss 150 groups (~50 min), hard/elite 500 (~35 min), easy 1000 (~3 min).
- Supplementary labelled dev cohort (devsupp, natural bucket 8 only, seeds 4): boss 40 groups, hard/elite 100. Primary comparison still uses fixed eval.txt; a second labelled train run adds devsupp to eval refs (dev-only rows, stop set unchanged).

## Stage 2 review (01:25)
- Gen 95 min: devsupp boss 2400 fights/19.1 min, devsupp hard/elite 12000/10.7, easy 40000/2.3 (1 death), hard/elite 30000/24.8 (974 deaths), boss 4500/37.8 (1996 deaths). Training runs ~2 min each, finite, deterministic (supp runs reproduce primary natural-dev numbers exactly).
- Outcomes (natural dev, augmented): boss Brier .164(s1)->.168(s2), elite .039->.039, HP MAE slightly better. Synthetic fixed dev: boss .176->.189, elite .052->.035. Supplementary dev (41 boss/60 elite donor clusters): boss .205->.179, elite .053->.047. Outcome gains from more data are modest/mixed; augmented best epoch dropped to 3 (early stopping on noisy stop NLL).
- Card effect dMSE vs zero (supplementary dev, larger): s1->s2 augmented boss -0.90±0.88 -> -1.42±0.86, elite -1.07±0.64 -> -2.32±0.62, hard +0.18±0.15 -> -0.26±0.16, easy 0 -> -0.18±0.06. Natural-only ~0 (can't learn card effects). Fixed-dev boss moved the other way (+0.56 -> +1.57±1.42; 12 clusters, noise). Reading: more synthetic data is starting to give real card-effect signal, clearest on elites.
- Diagnostic added: `diag/compare.py`. Training lr now overridable via CARD_OUTCOME_LR env (default .003 unchanged) in apps/combat_transition/train_marginals.py.
## Stage 3 plan (launched ~01:30, est. ~3.5h gen -> ~05:00)
- Same recipe (seeds 2, buckets 4-7), scaled: easy 1500 groups, hard/elite 1400, boss 2x275 (two chunks so a partial is usable). Boss weighted most (largest outcome error, biggest card effects).
- Labelled training-setting diagnostic: lr .001 variant on cumulative data (early stop at epoch 3 suggests lr/noise issue). Primary comparison stays at lr .003.
- lr .001 diagnostic (s2 data, supp eval): no gain (boss Brier .179->.183, card dMSE boss -1.23±.92, elite -1.92±.50, similar/slightly worse). Keep lr .003.
- 05:15 Training seed now overridable via CARD_OUTCOME_SEED (default 0 unchanged). Running seeds 1,2 for s1/s2/s3 supp to measure training-seed variance (s2->s3 card dMSE drop may be noise).

## Stage 3 review (05:30)
- Gen 3.6h: easy 60000 fights/3.6 min (11 deaths), hard/elite 84000/70.2 min (3162 deaths), boss 2x8250/71.5+71.8 min (3304+3389 deaths). Training ~4 min per run.
- Training-seed check (seeds 0,1,2; supplementary dev; table in diag/seedtable.md): seed0-only comparisons were misleading (s2 seed0 was lucky on card dMSE). Across seeds, augmented improves monotonically with data: boss Brier natural dev .170->.166->.155, synthetic boss .207->.182->.172, synthetic elite .054->.046->.046. Card dMSE vs zero (negative=better): elite +0.45 -> -1.38 -> -1.93 (all seeds <0 at s2/s3), boss +1.19 -> +0.06 -> -0.59, easy +0.14 -> -0.09 -> -0.13, hard ~0. Natural-only never beats zero on card effects.
- Decision: ~2.4h budget left. Boss/elite are still improving with data; spend ~75 min on a stage-3c extension (boss 180 groups, hard/elite 500, same recipe, new seeds) then retrain 3 seeds (~15 min), report by ~07:20. Easy/hard saturated -> no more easy.

## Stage 3c review / close (07:05)
- s3c gen: hard/elite 30000/26.7 min, boss 5400/47.0 min. 3-seed retrain: no further gain (boss Brier .172->.180, elite card dMSE -1.93->-2.06, within seed range). Data returns have plateaued for this fixed topology.
- Stopped here (~07:05, within 9h budget). REPORT.md written. Recommended next: capacity/topology sweep on cumulative data + dedicated high-seed card-effect eval set.
