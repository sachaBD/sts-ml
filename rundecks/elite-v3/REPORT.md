# elite-v3 report

**Status: pending (deck not yet run).** Plan: [PLAN.md](PLAN.md). Execution log: [LOG.md](LOG.md).

## Headline
<one paragraph: did v3 (value-only / policy search) beat MCTS on the elite mix? final test run or not, and why>

## Data
<stage 1 (and 4) fights, rows, skipped replays, potion / relic coverage from check_data>

## Training
<t1 / t2 (/ t3): best epoch, validation MSE, policy CE / top-1, keep MSE>

## Validation (450 bucket-5 fights, paired vs MCTS at 20k/8 on the current simulator)
<results/stage3a.md, stage3b.md, stage4.md tables>

## Final test (1,500 reserved fights)
<results/final.md, or why it was not run>

## Caveats
- Old (pre-change simulator) baselines are not comparable; MCTS was re-run here.
- Choosing the best of several arms on the same validation fights biases that arm's validation estimate upward.
- <skipped replays, anything that deviated from the plan>

## Next steps
<recommendations>
