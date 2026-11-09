# Champ full corpus

## Frozen partition preparation

`prepare.py` groups decks by card-ID multiplicity, ignoring upgrades. It reserves 950 unseen
compositions for validation and 1,000 for final testing; all 50 legacy compositions are excluded
from training. Equivalent held-out variants are withheld. Selection does not use fight outcomes.
Legacy evaluation retains ten unchanged historical starts per deck; new validation/final starts
use deterministic seeds and sampled potions.

```sh
PYTHONPATH=. .venv/bin/python experiments/champ-full-corpus/prepare.py \
  --pool runs/benchmark/megacrit-ironclad-a20-candidates-20261102/decks.jsonl \
  --legacy runs/benchmark/champ-50-inputs/frozen \
  --out runs/benchmark/champ-full-corpus-v1/frozen
```

The output directory must not exist. The original v1 partition contains 7,941 training loadouts,
950 validation decks, 50 legacy decks and 1,000 final decks. `selection.json`, `summary.json` and
`sha256.json` record membership, counts and input/output hashes. Final starts are reserved, not
routine evaluation inputs. These are the v1 inputs; v2 uses a separately frozen corpus.

## Legacy evaluation sets

`legacy_core` uses 50 historical fights for each of the 10 deeply studied decks.
`legacy_extended` retains 10 fights for each of the other 40 decks. Generate from the repo root:

```sh
PYTHONPATH=. .venv/bin/python experiments/champ-full-corpus/split_legacy.py \
  runs/benchmark/champ-full-corpus-v1/frozen/legacy.jsonl \
  runs/benchmark/champ-50-inputs/frozen/validation.jsonl \
  runs/benchmark/legacy-core50-extended10
```

The v1 and v2 frozen legacy files are byte-identical, so they share these inputs.
