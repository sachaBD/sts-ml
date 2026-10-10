# human_champ_bench_v1

Producer: `python -m apps.human_champ.bench prepare`. Contract: [schema.py](schema.py).

One retained human Ironclad A20 Champ deck, two seed-derived fight starts: numeric human
run seed (signed values reinterpreted as uint64), and a deterministic SHA256-derived fresh
seed. `deck_id` is human `play_id`; `fight_id` joins agent combat_v4 outputs.

Skip non-2020 builds, non-exact reconstructed decks, Runic Dome, unknown bottled cards,
unmapped names, Lizard Tail (usage unknown), Ritual Dagger / Genetic Algorithm (misc unknown),
and invalid HP/upgrades. Human outcomes and skip reasons are retained in `skipped.jsonl`.
Sampling is deterministic, across eligible decks without replacement.

Approximations: canonical sorted deck order; RNG streams begin from the seed rather than
historical counters. Cyclic relic counters reset to zero (`reset_counters`); Neow's Lament
assumed spent at Champ. Girya uses prior LIFT logs, Du-Vu Doll counts reconstructed curses.
Relic acquisition order follows the final record where recoverable. Previous room comes
from path_per_floor; no potions and zero gold (not used by current combat). All agents receive
identical starts. Historical balance differences are only noted, not corrected in v1;
`changed_cards` tags known Ironclad/colorless v2.2 changes, not a complete version diff.

The human comparator MUST use the retained paired decks. Exact-deck filtering heavily
selects for human losses; do not call this a population human skill estimate. Per-deck agent
scores average both seeds, and uncertainty uses decks, not independent fight starts.
