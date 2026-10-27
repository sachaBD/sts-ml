# Panache: 64 simulations, larger transformer

Approved 25 Oct 2026. Practical combined change, not a factorial comparison: user explicitly prefers a new low-budget run over repeats/controls.

## Question

Can a larger network with the current batched TensorRT inference path retain useful panache strength at 64 simulations, while improving wall-clock throughput?

## Amendment: 2,000-game updates (approved 25 Oct, ~15:18)

The initial 1,000-game update averaged 297/1,024 real leaves per GPU batch (29% fill), taking 79.16 s of gameplay plus 0.61 s evaluator setup. The bootstrap evaluation averaged 131 leaves (13% fill). Matching the old CPU run's game count was a poor fit for GPU batching.

User approved **2,000 games per update**. Preserve completed bootstrap and iteration 1; resume from iteration 1's model, optimizer and replay. Iterations 2–10 use 2,000 games, for **19,000 total new self-play games** (1,000 + 9 × 2,000). The interrupted iteration 2 was archived under the run's `interrupted/` directory, not included in replay. No bootstrap repeated. The old checkpoint lacked RNG state, so the resumed training sampling RNG restarts; future checkpoints save RNG states. This is not a bit-exact uninterrupted run.

The command below describes the original launch. For the continuation, use `--games 2000 --resume-from 1` in place of `--games 1000`. Do not launch a second process against the active output directory.

## Original recipe

- New transformer: width 128, 4 layers, 4 heads, pre-norm, standard full attention, existing tokens and policy/value heads. No specialized attention variant; sequences are short.
- Fresh weights, torch seed 0; same 300 stored MCTS20k bootstrap games, 2,000 training steps.
- 10 iterations of 1,000 games; 64 simulations for collection AND evaluation, 16 considered root actions, 8 belief particles, rollout mix 0.5, Gumbel scale 1 for collection / 0 for evaluation.
- 100 training steps per iteration, batch 2,048, AdamW lr 0.001, replay window 5 iterations. Same as prior recipe.
- 10 CPU workers, 200 concurrent game slots per worker, GPU batch 1,024, 48 tokens. Originally 1,000 games; superseded by the approved amendment above to fill the available slots.
- Same 300 held-out evaluation starts at iterations 0, 3, 10. Reuse the old run's training seed namespace for matching starts, without replaying old games.
- No additional control run. Compare descriptively with stored `nn-gumbel-panache-v1` (Gumbel256, width64/layers2, CPU inference): iteration 0/3/10 = 89/132/179 wins out of 300. Also stored PUCT2k = 174/300 and MCTS20k = 124/300.

## Timing and interpretation

- Per-iteration `play_seconds` includes evaluator setup; `train_seconds` includes data loading, training, saving and both ONNX exports.
- Native `worker.log` separately reports evaluator setup and gameplay wall seconds, plus GPU batches, mean leaves and GPU busy time.
- Compare end-to-end throughput and total wall time, not network-only evaluations/s or utilization alone.
- One training run, one deck. Paired seed intervals measure evaluation variation conditional on these trained networks, not training-seed uncertainty or unseen-deck generalization.
- Model capacity, search budget and inference path all change together. Success/failure describes this combined setup; it does not identify causal contributions.
- If TensorRT export/build or numerical behavior fails, stop and diagnose rather than silently change settings.

## Command

```sh
.venv/bin/python -m agents.combat.nn.loop \
  --start experiments/25-10-26/a-gumbel/nn/panache_start.json \
  --bootstrap experiments/25-10-26/a-gumbel/nn/panache_mcts_bootstrap.jsonl \
  --test experiments/25-10-26/a-gumbel/nn/panache_test.jsonl --test-at 0,3,10 \
  --out runs/schema=combat_v4/date=2026-10-25/id=nn-gumbel64-panache-128x4-v1 \
  --dim 128 --layers 4 --heads 4 --sims 64 \
  --iterations 10 --games 1000 --steps 100 --bootstrap-steps 2000 \
  --batch 2048 --workers 10 --gpu-batch 1024 --gpu-tokens 48 \
  --fight-namespace id=nn-gumbel-panache-v1
```

Output log: `runs/schema=combat_v4/date=2026-10-25/id=nn-gumbel64-panache-128x4-v1.log`.
Live progress: run directory's `status.json`; `dashboard.py RUN_DIR` displays progress and paired comparisons against stored references.
