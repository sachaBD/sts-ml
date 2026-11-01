# Champ 25-deck overnight run

Warm-start checkpoint 132 (entity 192/4/6), keeping the original ten decks and adding
15 diverse decks with historical MCTS wins between 4/20 and 15/20; see [selection](DECKS.md).
Freeze 50 validation starts/deck: retain the original 500; each added deck has 20 recorded
MCTS starts plus 30 fresh starts. These are monitoring starts, not a held-out test.

Settings: 64 simulations, eight particles, 50/50 NN/guided-rollout leaves; noise 1 in
self-play, 0 in validation. 5,000 fights/update, 100 optimizer steps, batch 2048, lr .001;
deck-balanced replay, five batches, 500k-row cap. Validate initially/every three updates.
No time limit. Bounded CUDA recovery (two retries) and 10 GiB disk reserve remain enabled;
checkpoints accumulate, so the reserve can eventually stop the run despite replay pruning.

Uses the current app/worker, not the obsolete loop; its PR #18 rollout/root semantics
differ from the old training run. Entity code/export was carried over unchanged from
the topology worktree; checkpoint architecture and native token contract stay compatible.

From `/home/sborowsk/project/sts/worktrees/champ-25-overnight`:
```sh
./apps/gumbel_nn_combat_expert_iteration/run.sh experiments/2026-10-31-champ-25/train.toml
# status/stop: same script with status RUN_DIRECTORY or stop RUN_DIRECTORY
```

Inputs/provenance: `runs/benchmark/champ-25-inputs/` (including selection source files).
Regenerate offline with `python -m experiments.2026-10-31-champ-25.prepare_inputs MAIN_CHECKOUT`.
Cleanup reclaimed 9.43 GiB of old replay/samples/caches; manifest at
`runs/benchmark/cleanup-20261031.json`. All old models/results/logs were retained.

Qualification passed: 16 owner tests; exact warm-start weights; 50 real GPU fights
across all 25 decks with no caps; two optimizer steps produced finite changed weights
and all four ONNX exports. Qualification run: `champ25-qualification-20261031`.

Active run: `runs/schema=gumbel_nn_combat_v1/date=2026-11-01/id=gumbel-nn-champ25-64s-mix50-20261031`.
Dashboard: http://127.0.0.1:8743/ ; launcher log: `runs/benchmark/champ25-overnight-launch.log`.
Verified live after the initial 1,250-fight validation and two complete 5,000-fight
updates (zero caps); iteration 3 underway. First two updates took 269 and 261 seconds,
including training/export, excluding scheduled validation. Longer-run variability unknown.
