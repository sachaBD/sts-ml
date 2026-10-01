# Combat-outcome predictor

A shared pre-combat predictor, not an agent: estimates P(win) and a final-HP distribution from persistent game state and encounter identity.

- `combat_outcome_v1.py`, `combat_outcome_v2.py`: versioned model architectures.
- `learn.py`: loading, batching, baseline, learning, and evaluation.
- `learn_marginals.py`: natural/augmented cohorts, frozen architecture loading, fitting, and paired evaluation.
- `architectures/`: named specs and `FROZEN.tsv`; once used, a spec/kind must not change behavior.
- `test_card_outcome_training.py`: colocated adapter, split, and fitting checks.

CLI entrypoints remain `apps/combat_transition/train.py` and `train_marginals.py`. GUI and agents import this owner directly, rather than importing another application's implementation. Checkpoints retain their original kind/args/state_dict representation; generated outputs remain in `runs/`.
