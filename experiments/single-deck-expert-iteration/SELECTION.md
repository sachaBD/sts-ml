# Initial deck selection check

Deck `7a9ada48-a4d6-41fc-a85b-0147618df00a`: fixed 19-card Barricade+/Entrench+/Body Slam+
loadout from the human-derived benchmark. HP 41/75, original fixed relics/counters,
no potions, Ironclad A20 Champ. No HP or loadout augmentation in this check.

MCTS20k played 20 fresh SHA256-derived selection seeds, excluding its two existing
benchmark seeds. Frozen native worker copied by human-combat-r01; max eight workers.
All plays completed and passed the worker's replay validation.

**18/20 wins = 90%. 95% Wilson interval: 69.9%–97.2%.**
No errors/caps. These are independent seed-derived encounters conditional on this fixed
simulator loadout, not human encounter replays. Do not count them as future monitoring,
final test, or training seeds. Previous 2/2 wins are selection history, not pooled into
the reported fresh-sample estimate.

Interpretation: winning is clearly achievable and MCTS appears strong at this HP.
The sample remains small and was collected after choosing a promising deck. There is
limited apparent win-rate headroom above MCTS here, but matching it at lower learned
search cost remains a meaningful task. A lower-HP selection check can establish a
more challenging band before freezing training/evaluation HP support. No further
selection or training was launched by this check.

Artifact: `runs/schema=combat_v4/date=2026-10-05/id=single-deck-mcts20/`.
Starts: `out/starts.parquet`; plays/actions/annotations: `out/play/`; summary: `out/summary.json`.
Reproduction wrapper: `experiments/single-deck-expert-iteration/screen.py` (reuses benchmark play).
