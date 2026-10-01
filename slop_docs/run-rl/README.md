# Real-run RL for card picks (started 2026-09-30)

**Direction:** learn card picks from **real** Act 1 runs (MCTS plays every fight). No simulator or outcome model in the
loop, so there is nothing for the policy to exploit. Progress = real clear rate on fixed eval seeds vs SimpleAgent picks
on the same seeds. Simulator / model numbers are diagnostics only.

The surrogate approach (`slop_docs/card-policy-loop/`, `apps/card_search/`) is untouched and can be revisited.

## What is trained
One network, V(run state) = expected score from here under the current policy (kind `run_policy_v1`, value head only).
Card pick = argmax over after-states (deck + card A / B / C, or unchanged for skip). Everything else: SimpleAgent.

## Reward (score of a finished run, in [0, 1])
- Act cleared: 1 (optionally `1 - hp + hp * HP_end / maxHP` with `--hp`, default 0).
- Death on floor f: `progress * f / 16` (`--progress`, default 0.25), so dying at the boss scores 0.25, on floor 8 0.125.
- Why not elites beaten: pathing is fixed (SimpleAgent), so elite count is mostly not the picker's doing.
- Density comes mainly from TD bootstrapping: every state gets a target from V of the next state (HP, deck after each
  fight), not only from the end-of-run outcome.
- The reward is applied at training time from logs: changing it only needs retraining, not new runs.

## Algorithm (`apps/run_rl/loop.py`)
1. Baseline: SimpleAgent picks on the eval seeds (through the same harness).
2. Iter 0 data: SimpleAgent picks + 20% random picks, on fresh training seeds.
3. Train V: nodes = start state, state after each fight, after-state of each pick. Target = TD(λ=0.7) return
   (iter 0: Monte Carlo). Loss = BCE. Data = the last 3 batches. Early stopping on 10% held-out seeds.
4. Next batch: greedy on V with 10% random picks. Retrain (initialised from and bootstrapping on the previous V).
5. Each iteration: greedy eval on the same eval seeds -> `curve.tsv` (net vs SimpleAgent, paired difference ± SE).

## Harness facts (measured 2026-09-30)
- `run_rl_worker` (`apps/run_rl/worker.cpp`): **6.6 worker-s per run** (22-run smoke), i.e. about 6,000 runs/hour on 11
  workers.
- 18/22 seeds reproduce the 09-27 baseline exactly with SimpleAgent picks; 4 differ in fight outcomes (the no-recording
  fight path or simulator changes since). So compare only against SimpleAgent run through this harness.
- Eval noise: 500 paired seeds give SE ≈ 2 points on the clear-rate difference.

## Files
`apps/run_rl/{worker.cpp, play.py, train.py, loop.py, common.py}`; CMake target `run_rl_worker`. Runs in `runs/run_rl/`.
