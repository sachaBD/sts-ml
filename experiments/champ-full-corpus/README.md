# Legacy evaluation sets

`legacy_core` uses 50 historical fights for each of the 10 deeply studied decks.
`legacy_extended` retains 10 fights for each of the other 40 decks. Generate from the repo root:

```sh
PYTHONPATH=. .venv/bin/python experiments/champ-full-corpus/split_legacy.py \
  runs/benchmark/champ-full-corpus-v1/frozen/legacy.jsonl \
  runs/benchmark/champ-50-inputs/frozen/validation.jsonl \
  runs/benchmark/legacy-core50-extended10
```

The v1 and v2 frozen legacy files are byte-identical, so they share these inputs.

## Historical comparison

Collect the PUCT/MCTS20k references and Gumbel specialist checkpoint on those same Core fights:

```sh
SPECIALIST=runs/schema=gumbel_nn_combat_v1/date=2026-11-01/id=gumbel-nn-champ50-64s-decay30-r2-20261101/out/training/test/iter084/results.jsonl
PYTHONPATH=. .venv/bin/python experiments/champ-full-corpus/legacy_baselines.py \
  runs/benchmark/legacy-core50-extended10/legacy_core.jsonl \
  runs/benchmark/champ-50-inputs/frozen/baselines.json "$SPECIALIST" \
  runs/benchmark/legacy-core50-extended10/baselines.json
```

New runs freeze this file via `evaluation.baselines`. For an existing run, pass
`--baselines runs/benchmark/legacy-core50-extended10/baselines.json` to the web dashboard.
This does not change the run's frozen inputs. The table can show all recorded fights or only
starts shared by every column; counts and fight-level Wilson intervals remain explicit.
