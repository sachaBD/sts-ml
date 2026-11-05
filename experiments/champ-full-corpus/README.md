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
