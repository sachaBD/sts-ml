# Slime Boss value model v1 results

## Scope

This experiment tests whether a value network trained from random-rollout MCTS
can guide search successfully in the same combat family.

- Scenario: Ascension 1 Slime Boss.
- Character: Ironclad at 80/80 HP.
- Deck: starter deck plus Bash+, Flame Barrier, Combust, Hemokinesis+, and
  Battle Trance.
- Encoding: public-information schema v2.
- Value model: width-64 permutation-invariant Deep Sets network.
- Search: determinization-based MCTS. Neural search replaces a newly expanded
  nonterminal leaf's random rollout with the network value. Terminal leaves
  always use the exact combat return.

## Bootstrap dataset

The accepted dataset is
`data/mcts-slime-v2-bootstrap-1000x500/mcts_slime_v2.parquet`.

| Property | Value |
|---|---:|
| Seeds | 1-1000 |
| Episodes | 1,000 |
| Decisions | 28,495 |
| MCTS simulations per decision | 500 |
| Teacher wins | 986 (98.6%) |
| Mean decisions per episode | 28.495 |
| Parquet SHA256 | `8371784eb72c0ba91496f8e812a67273fc14f8a11b9580017fb65c678f44aecc` |

The writer validates rows while streaming, writes bounded Parquet row groups to
a partial file, and atomically promotes the shard only after validation.

## Value prediction

Episodes were split deterministically into 800 training and 200 validation
episodes, with no episode overlap. The model trained for 20 epochs against
`mcts_value`.

| Held-out metric | Model | Train-mean baseline |
|---|---:|---:|
| MSE | 0.011819 | 0.531975 |
| MAE | 0.064041 | - |
| Pearson correlation | 0.989118 | - |

The hardest evaluation slice was the boss at 40-60% HP: MSE 0.035591, MAE
0.132311, and Pearson correlation 0.917709. This remained substantially better
than that slice's 0.258004 baseline MSE.

Checkpoint:
`runs/slime-v2-value-first/value_checkpoint.pt`

Checkpoint SHA256:
`83690c0f89f5f60fd8c968378e1d04e0ed2d88705b04ef4f9c1725b5a23ca289`

## Held-out gameplay

The first gameplay comparison used seeds 2001-2200, which are outside both the
training dataset and the separate generation preflight. Both agents received
100 MCTS simulations per player decision.

| Metric | Neural-leaf MCTS | Random-rollout MCTS |
|---|---:|---:|
| Wins | **200/200 (100%)** | 195/200 (97.5%) |
| Mean final HP | **46.475** | 33.385 |
| Median final HP | **47** | 33 |
| Mean split HP | **41.345** | 44.869 |
| Median split HP | **43** | 46 |
| Mean decisions | **29.570** | 31.255 |
| Mean wall time per fight | 1.866 s | 0.071 s |

On paired seeds, neural MCTS won all five fights lost by rollout MCTS and lost
none. It retained 13.09 more HP on average and split the boss about 3.41 HP
lower when both fights reached a split. The Wilson 95% lower bound for 200 wins
in 200 trials is approximately 98.1%.

The gameplay results and incremental log are stored at:

- `runs/slime-v2-gameplay-200x100/episodes.jsonl`
- `runs/slime-v2-gameplay-200x100/evaluation.log`

`data/` and `runs/` are intentionally ignored by Git; the hashes and aggregate
results above are the durable record.

## Interpretation and limitations

This establishes that the learned values are useful for action selection, not
merely correlated with teacher labels. At equal simulation count, neural MCTS
won every held-out fight and produced healthier wins and lower splits.

The result is deliberately narrow:

- one fixed enhanced deck and one starting HP;
- one encounter and ascension;
- a highly win-skewed teacher dataset;
- training examples primarily from the teacher's chosen trajectories;
- no policy head;
- synchronous Python/C++ inference, approximately 26 times slower than rollout
  MCTS in this comparison;
- no paired 500-simulation rollout comparison on these held-out seeds.

The next learning milestone should test generalization across a controlled
family of feasible decks and starting combat conditions before adding a policy
head or optimizing inference.
