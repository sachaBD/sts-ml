# Legacy evaluation sets

`legacy_core` contains the 10 deeply studied decks; `legacy_extended` contains the other 40.
Both retain the original 10 fights per deck. Generate the inputs from the repo root:

```sh
PYTHONPATH=. .venv/bin/python experiments/champ-full-corpus/split_legacy.py \
  runs/benchmark/champ-full-corpus-v1/frozen/legacy.jsonl runs/benchmark/legacy50-split
```

The v1 and v2 frozen legacy files are byte-identical, so they share this split.
