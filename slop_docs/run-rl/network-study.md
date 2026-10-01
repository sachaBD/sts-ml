# Network study (run value V for card / rest / path decisions)

**Question:** would a bigger or better network make better decisions?
**Short answer so far:** not detectably. On the ~38k real runs we have, every reasonable network predicts outcomes
equally well. The map input is the only part that clearly matters. The limit is data (each run is one noisy outcome),
not network size.

## How it was tested
- Same data for every network: the 38,000 v1 training runs (`run_rl_v1/2026-09-30/v1`, iters 0-18), 15% of seeds held
  out. Target: the run's final score (1 = clear, else 0.25·floor/16): fixed, unlike the loop's TD targets.
- Metric: held-out binary cross-entropy (lower = better). Predicting the average for everything scores 0.5255.
- New topology kind `run_policy_v2` (`python/sts_combat_rl/topology/run_policy_v2.py`): run_policy_v1 plus options for
  width / depth / dropout, **deck attention** (cards in the deck attend to each other, so synergies like
  Limit Break + Inflame can be represented) and **no map**. Script: `apps/run_rl/netstudy.py`; results
  `scratch/netstudy/screen{1,2}.jsonl`.

## Results (held-out BCE; 2 training seeds where shown)
| network | params | BCE | reading |
|---|---|---|---|
| small (v1 size: width 32, hidden 64, dropout 0.3) | 66k | 0.4623 / 0.4658 / 0.4639 | reference |
| small + deck attention | 75k | 0.4668 / 0.4644 | same |
| mid (64 / 128, dropout 0.5) | 206k | 0.4644 / 0.4645 | same |
| mid + attention (48 / 128 ×2, dropout 0.4) | 205k | 0.4692 / 0.4647 | same |
| wide (64 / 256 ×2, dropout 0.1) | 496k | 0.4698 | overfits after 1 epoch |
| wide + attention | 529k | 0.4644 | attention offsets some overfitting |
| wide, no map | 424k | 0.4727 | worse |
| small, no map | 52k | 0.4704 / 0.4708 | **worse: the map matters** |

Training-seed noise is ~±0.002, so everything from 0.462 to 0.467 is a tie.

## What this means
- **Data-bound, not capacity-bound.** Bigger networks only fit the noise in 38k coin-flip outcomes sooner.
- **The map encoding earns its place** (~0.006 worse without it, consistent over seeds and sizes). That fits the v2
  result: path choice turned out to be a big lever.
- **Caveat:** held-out BCE measures "how well does V predict outcomes", not "does V rank the options at a decision
  correctly". A network could tie here and still pick better. Only real games can show that, and small differences
  (< ~3 points) are below what a 500-seed real test can detect.

## Next (if pursued)
- Real-game check of one alternative (small + attention or mid) against the reference on the same data and seeds,
  only if the loop plateaus with spare compute.
- The more promising lever is more signal per run: e.g. auxiliary targets (HP lost per fight, HP at the boss) that
  every run provides densely. Untested.
