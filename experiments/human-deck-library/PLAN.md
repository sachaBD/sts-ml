# Human fixed-loadout library v1

Goal: a small diverse library of Ironclad A20 Champ combat starts, with MCTS20k
seed success estimates, exact reconstructed starts and complete action replays.
Not a representative human population study, not a precise deck ranking.

20 loadouts × 20 selection seeds:
- Two existing references: Barricade at 41/75 HP and Flex/JAX/Limit Break at 33/80 HP.
  Reuse their completed 20-seed MCTS screens (40 plays); native worker checksums match.
- 18 additional human decks: block 3, Demon Form 3, exhaust 4, strength 4, mixed 4.
  Archetype buckets are descriptive, mutually exclusive by priority, not exact strategic labels.
- Within each bucket, greedy diversity over non-starter card sets, deterministic hash tie-break.
  Unique card-count/upgrade/misc families; manifests frozen before outcomes. No outcome filtering.
- Original reconstructed HP and fixed relic/counters per loadout; no potions. No HP augmentation
  or historical balance correction. Report HP explicitly: library rows represent loadouts,
  not deck strength independent of HP. Existing reconstruction caveats remain.
- 360 new MCTS20k fights, 10 workers. No neural-model training/evaluation in this pass.
- Independent namespace-derived selection seeds, excluding source and reference seeds.
  These are discovery/selection data, not an untouched final set for later experiments.
- Report wins over completed seed trials with 95% Wilson intervals and explicit status counts.
  n=20 is deliberately coarse; retain losses and errors, do not silently filter decks.

Sources: original 431-deck benchmark plus expanded 1,133-deck reconstructed human pool.
Runner: `python -m apps.human_champ.library run` reuses the existing managed combat play stage.
Raw records: `runs/schema=combat_v4/date=2026-10-05/id=human-deck-library-v1/`.
Outputs: `out/manifest.json`, `catalogue.json`, `catalogue.csv`, `REPORT.md`, exact
`starts.parquet`, `new-starts.parquet`, reference results and new `play/` action replays.
All native inference uses the frozen worker copied from the final specialist comparison.

Overall dashboard (from repo root):

```sh
watch -n 5 '.venv/bin/python -m apps.human_champ.library status --run runs/schema=combat_v4/date=2026-10-05/id=human-deck-library-v1'
```

Next: evaluate a selected learner on these same starts to find teacher-recoverable learner
weaknesses, then choose a contrasting second task for joint training. MCTS success alone
cannot establish theoretical winnability or learner difficulty.
