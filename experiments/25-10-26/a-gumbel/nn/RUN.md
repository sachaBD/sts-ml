# nn-gumbel-panache-v1 — registration (APPROVED; completed)

**Question:** starting from MCTS games, does Gumbel self-play with the new transformer reach the stored 2k-sim network's
strength on panache (174/300 test wins; MCTS20k 124/300)?

## Run

| stage | what | time (estimate) |
|---|---|---|
| 0. Bootstrap | Fresh network trained on the 300 stored MCTS20k panache games (`champ-ten-rollout-v1` bootstrap; no replay). Value target = game result, policy target = MCTS visit counts. 2,000 steps, batch 2,048. | ~1 min |
| 1–10. Self-play | Each iteration: 1,000 games, Gumbel 256 sims, rollout mix 0.5, Gumbel noise 1, 8 hidden-state guesses. Then 100 training steps (batch 2,048, lr 1e-3) on the last 5 iterations' games (the MCTS games drop out after iteration 5). | ~3.5 min each, ~35 min total |
| Test | 300 fixed panache test seeds, greedy (Gumbel 256, no noise), at iterations 0, 3 and 10. | ~1 min each |

Total ≈ 40 min, 10 workers.

- Network: transformer, 2 layers, width 64, 2 heads, ~695k parameters (~105k excluding embedding tables).
- Iteration 3 ≈ equal self-play CPU to the stored 2k-sim run (~3,000 games × ~2 s vs 1,200 × 5.5 s); iteration 10 ≈ 3×.
- Test results compared, paired by seed, with the stored 2k-sim network (174/300) and MCTS20k (124/300).

## Outputs

`runs/schema=combat_v4/date=2026-10-25/id=nn-gumbel-panache-v1/`: `iterNNN/{results.jsonl, samples-*.bin, model.pt, model.onnx}`,
`log.jsonl` (one line per iteration), `test/iterNNN/results.jsonl`, `status.json` (live phase/progress).

Dashboard: `watch -n 10 .venv/bin/python experiments/25-10-26/a-gumbel/nn/dashboard.py` shows the stage and iteration
(n/10), the current phase with games played / training step, minutes per iteration and ETA, per-iteration self-play
win rate, and the test results so far against 174/300 and 124/300.

## Work before launch (~45 min)

1. `nn_play --import`: replay stored MCTS games into training samples (visit counts as the policy).
2. Loop: bootstrap stage, test checkpoints, live progress (`status.json`, results appended as games finish).
3. Dashboard script.
4. 1-minute smoke run of the whole thing at tiny sizes.

## Caveats

- One run; 300 test seeds give a roughly ±7 pp paired CI.
- The stored reference network was trained on ten decks; this one only on panache.


## Completed result

Run completed with 10,000 new self-play games. Greedy tests on the same 300 panache seeds:

| iteration | wins | vs stored 2k-sim network (174/300), pp [95% CI] | vs MCTS20k (124/300) |
|---|---:|---:|---:|
| 0 | 89/300 | −28.3 [−34.4, −22.2] | −11.7 [−17.3, −6.0] |
| 3 | 132/300 | −14.0 [−20.3, −7.7] | +2.7 [−3.7, +9.0] |
| 10 | 179/300 | +1.7 [−5.2, +8.5] | +18.3 [+11.8, +24.9] |

Conditional paired normal-approximation intervals; one training run, one deck. Stored reference learned ten decks.
This shows comparable final play strength with 256 simulations, not superior training compute efficiency.
This run used CPU inference. Later GPU throughput work is documented in `agents/combat/nn/readme.md`.

Generate the local input files with `prepare_panache.py`; they are not checked into git. The benchmark scripts
are development measurements, not approved training launchers. No 64-simulation training run has been launched.
