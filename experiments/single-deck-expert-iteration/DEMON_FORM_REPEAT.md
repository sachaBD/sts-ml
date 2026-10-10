# Original-recipe repeat: Demon Form / Runic Pyramid

User approved 2026-10-06. One fixed deck, no corpus expansion or strategy hints.

- Deck: `249e5246-153d-4030-bc11-598774146147`; A20 Champ, original 34/52 HP, relics/counters, no potions.
- Selection: MCTS20k 14/20 (70%, 95% Wilson 48.1–85.5%); selection trajectories are not training or final-test data.
- Fresh width64 sigmoid policy/value network, torch seed 0; same frozen native worker as A.
- 200 fresh MCTS20k bootstrap fights, 10 epochs lr=1e-3.
- Five updates ×100 fresh learner2k fights, exploration on; 3 epochs/update lr=3e-4, optimizer resumed, grad clip 1.
- Actual win value targets; root-visit policy targets with concentration weighting; 64 sampled states/fight/epoch; last ten learner batches, teacher taper 200→120 over these updates.
- 100 fixed monitoring seeds: evaluate bootstrap and update five, same MCTS20k reference. No monitoring gradient updates.
- 600 fresh final seeds reserved, NOT automatically played. Review bounded training result first.
- 10 gameplay workers (original A used 8; user preference for new runs is 10).
- Source: corpus pool human start; prior library selection starts excluded. New namespace `demon-form-fresh-v1`.
- Minimal runner extension: configurable deck ID and deck-dependent HP/report labels, no recipe changes. Existing default-deck resume configs remain compatible.
- Preflight: all 20 selected-deck teacher fights encoded/replayed successfully with frozen worker; six single-deck unit tests passed. Worker SHA256 matches A final frozen worker.

Run: `runs/schema=combat_v4/date=2026-10-06/id=single-deck-demon-form-v1/`

```sh
watch -n 5 '.venv/bin/python -m apps.run_rl.single_deck_status --run runs/schema=combat_v4/date=2026-10-06/id=single-deck-demon-form-v1'
```

Logs: run `logs/stdout.log`, `logs/stderr.log`; detailed stage logs under `out/logs/` and `out/iter*/logs/`.

## Continuation approved

User requested ten additional rounds, extending horizon from update 5 to update 15,
with unchanged teacher taper/replay/training settings and 10 workers. Resume helper:
`apps.run_rl.resume_single_deck --run <run> --updates 15`. Managed metadata records
continuation and runtime status; existing compute markers skip completed stages.
Monitoring remains every five updates (10 and 15). Final seeds stay untouched.

Added `out/training-win-rate.png` and JSON: one point per round, new exploratory
batch plus active replay set including remaining teacher fights. Fight-count Wilson
95% bounds are descriptive: replay overlaps and mixes policies, not a held-out
performance estimate. PNG replaced atomically after each round. Existing first five
rounds backfilled; seven focused tests passed and actual resume verified at collection 6.
