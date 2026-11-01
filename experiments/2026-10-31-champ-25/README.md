# Champ 25-deck experiment (historical)

Warm-started from the ten-deck model at checkpoint 132. This is a record, not a maintained training preset.

- Run: `runs/schema=gumbel_nn_combat_v1/date=2026-11-01/id=gumbel-nn-champ25-64s-mix50-20261031`.
- Frozen inputs, selection provenance and hashes: `runs/benchmark/champ-25-inputs/`.
- Launch revision: `c32d267cf5764be936fe484f33f1d115de3da96d`.
- Stopped cleanly after iteration 135 because of the disk reserve.

Artifact paths are relative to the original run checkout; artifacts are not committed here.
`train.toml` and `prepare_inputs.py` retain the original settings and machine-local paths.
See [DECKS.md](DECKS.md) for selection references. Validation was used for monitoring, not held-out testing.
