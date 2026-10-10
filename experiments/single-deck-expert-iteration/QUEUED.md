# Later deck checks

## Flex / JAX / Limit Break / Heavy Blade+

User reference: `date=2026-10-05`, `human-champ-v1-pv`, corrected shorthand **`002005aa1`**
(original message: `00505aa1`).
Request: remember and check MCTS later; do not interrupt or replace the current fixed
Barricade experiment.

The corrected shorthand has one extra zero compared with the closest stored ID.
Probable intended match:
`human-champ:00205aa1-6004-4ab0-8600-655bddec017e:{human,fresh}` (HP 33/80), which contains
Flex, JAX, Limit Break and Heavy Blade. Another deck with those cards is
`66761770-9751-4677-b9d4-24909985fc02` (HP 48/80). Confirm the viewer link/short ID before
launching the later MCTS screen. Keep the user's original reference rather than silently
substituting the probable match.

Status: MCTS selection check completed on the matching stored deck
`00205aa1-6004-4ab0-8600-655bddec017e` (16 cards, Flex+/JAX+/Limit Break+/Heavy Blade+).
20 fresh seeds, fixed 33/80 HP, fixed relics/counters, no potions, MCTS20k, 10 workers.
**18/20 wins (90%); 95% Wilson interval 69.9–97.2%.** All plays completed; no errors/caps.

Artifact: `runs/schema=combat_v4/date=2026-10-05/id=strength-deck-mcts20/`.
Original two benchmark seeds excluded; these are selection seeds, not a final test.
Next possible check: the Barricade specialist on these identical starts, to measure
transfer before deciding whether to train jointly. That learned-model check has not run.
