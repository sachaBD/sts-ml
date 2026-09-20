# Slime Boss neural-search roadmap

## Completed

1. Public semantic state/action encoding in C++.
2. Slime Boss scenario and multi-monster evaluation.
3. C++ state dump and PyTorch Deep Sets smoke pass.
4. Variable-token training batches with segmented sum pooling.
5. Validated streaming MCTS records and immutable Parquet output.
6. Bootstrap value training with an episode-level held-out split.
7. Neural leaf evaluation inside determinization-based MCTS.
8. Held-out gameplay comparison against equal-budget rollout MCTS.

The v1 model won 200/200 unseen fixed-deck fights at 100 simulations per
decision. See [results_v1.md](results_v1.md) for the dataset, value metrics,
gameplay results, hashes, and limitations.

## Next

### 9. Controlled scenario generalization

- Define a reproducible distribution of feasible Ironclad decks and starting
  combat conditions.
- Preserve scenario parameters in every episode's provenance.
- Stratify train, validation, and gameplay evaluation by deck rather than only
  by combat seed.
- Establish coverage and minimum-performance gates for weak, average, and
  strong decks and for relevant starting-HP bands.
- Generate MCTS labels over this controlled distribution and measure
  out-of-deck as well as in-distribution generalization.

### 10. Iterative neural search

- Generate on-policy records from neural-guided MCTS.
- Mix bootstrap and on-policy examples to avoid abrupt distribution collapse.
- Retrain and compare against fixed held-out scenario suites.

### 11. Policy head

- Add semantic action tensors and masks.
- Train against MCTS visit distributions.
- Use learned priors with PUCT.

### 12. Inference optimization

- Remove the synchronous subprocess round trip after the learning design is
  stable.
- Batch leaf inference or adopt an in-process model runtime.
- Compare decisions as well as outcomes before and after optimization.

## Slime Boss acceptance signal

For every evaluation family, log boss HP before the triggering attack, split
HP, child HP, player HP at split, final outcome, and final player HP. The
learned agent should prefer lower or better-timed splits when immediate damage
is strategically worse.
