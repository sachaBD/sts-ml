# run_rl v1 check-ins (runs/run_rl/v1)

Eval = greedy net picks on 500 fixed seeds (600000000000..499), paired with SimpleAgent picks on the same seeds
(61.2%). ± = 1 SE of the paired difference. The same eval seeds are reused each iteration: confirm on fresh seeds
before claiming a final number.

## Check-in 1 (~2 h after start)
| iter | net clear | diff vs simple | train-batch clear (10% explore) |
|---|---|---|---|
| 0 | 65.4 | +4.2 ± 2.6 | 55.4 (SimpleAgent + 20% random) |
| 1 | 73.2 | +12.0 ± 2.3 | 58.4 |
| 2 | 73.4 | +12.2 ± 2.4 | 66.0 |
| 3 | 76.0 | +14.8 ± 2.4 | 68.5 |
| 4 | 75.8 | +14.6 ± 2.4 | 67.1 |
- Per boss (simple -> iter 4): Hexaghost 51 -> 70, Slime 61 -> 81, Guardian 70 -> 75 (~150-180 seeds each).
- Reached boss 90% -> 92%; most of the gain is at the boss.
- Net skips 18% of rewards, agrees with SimpleAgent on ~46% of picks.
- Open question: train batches clear ~8 pts below eval. Exploration cost or easier eval seeds? Fresh-seed confirmation
  will tell (planned after the loop).
- No plateau yet; window stays at 3 batches.

## Check-in 2 (~4 h): plateau -> wider lookback
| iter | net clear | diff vs simple | train-batch clear |
|---|---|---|---|
| 5 | 75.6 | +14.4 ± 2.4 | 68.9 |
| 6 | 79.2 | +18.0 ± 2.4 | 72.1 |
| 7 | 77.0 | +15.8 ± 2.3 | 68.8 |
| 8 | 75.2 | +14.0 ± 2.4 | 71.3 |
- Flat at ~76-77% since iter 3 (per-iteration SE of the clear rate ~1.9 pts).
- Per the agreed rule: restarted from iter 9 with window 3 -> 8 batches (16k runs) and recency weight 0.85 per batch
  (oldest batch 0.32). `loop.py --window 8 --decay 0.85`.
- Training made faster first (numpy encoding, batched GPU TD targets, encode once per fit): 16k runs train in ~4 min
  (was 8-16 min for 6k). Play step is now resumable.

## Check-in 3 (~6 h): still flat after widening
| iter | net clear | diff vs simple | train-batch clear |
|---|---|---|---|
| 9 | 77.4 | +16.2 ± 2.3 | 69.9 |
| 10 | 77.4 | +16.2 ± 2.4 | 69.3 |
| 11 | 77.0 | +15.8 ± 2.4 | 71.4 |
| 12 | 75.4 | +14.2 ± 2.4 | 70.9 |
- Window 8 / decay 0.85 shows no visible change yet (the window is still filling with recent batches).
- Where runs end (iters 9-12 pooled, 2,000 eval runs, vs SimpleAgent 500): death before boss 10.0% -> 8.4%, death at
  boss 28.8% -> 14.9%. Hexaghost 51 -> 68%, Slime 61 -> 82%, Guardian 70 -> 79%. HP after a won boss 32.7 -> 34.0.
- The picker roughly halved boss deaths; remaining losses are split ~1:2 between pre-boss (mostly elites) and boss.

## Check-in 4 (~8 h): plateau holds at ~76-77%
| iter | net clear | diff vs simple |
|---|---|---|
| 13 | 74.6 | +13.4 ± 2.4 |
| 14 | 77.4 | +16.2 ± 2.4 |
| 15 | 76.4 | +15.2 ± 2.5 |
- Mean of iters 3-15: ~76.5%. Window 8 + decay made no detectable difference. Letting the loop finish (iter 19), then a
  fresh-seed confirmation of the final model (not the best-on-eval one, to avoid selection bias).
