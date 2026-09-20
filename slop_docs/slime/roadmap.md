# Slime Boss neural-search roadmap

## Completed

1. Public semantic state/action encoding in C++.
2. Slime Boss scenario and multi-monster evaluation.
3. C++ state dump and PyTorch Deep Sets smoke pass.

## Next

### 4. Training tensor encoding

- Convert semantic tokens to the feature vectors in `encoding.md`.
- Keep card and monster sets variable length.
- Batch with token-to-state indices and segmented sum pooling.
- Use the same field order and scaling in dataset generation and inference.

### 5. MCTS training records

- Return root visit counts and Q values from MCTS.
- Record one decision per row.
- Write immutable Parquet shards with PyArrow.
- Store MCTS configuration and dataset provenance in a small manifest.

### 6. Bootstrap value training

- Generate teacher episodes with MCTS.
- Measure win rate before adding curriculum.
- Train the value model first on MCTS root values.
- Retain terminal outcomes for comparison and later target changes.

### 7. Neural-guided MCTS

- Replace leaf rollouts with value inference.
- Compare against the same MCTS budget and held-out seeds.
- Regenerate data from the improved search.

### 8. Policy head

- Add padded semantic action tensors and masks.
- Train against MCTS visit distributions.
- Use policy priors in PUCT.

## Slime Boss acceptance signal

Log boss HP before the triggering attack, split HP, child HP, player HP, final outcome, and final player HP. The learned agent should prefer lower or better-timed splits when immediate damage is strategically worse.
