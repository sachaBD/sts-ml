# Joint expert iteration — 2026-10-01

Goal: strongest autonomous full-game StS agent feasible on one machine. Tonight: improve Act-1 combat and overworld decisions together; schemas/replay are instrumental, not the objective.

Approved: staggered updates; easy guided rollout 500, hard neural leaves 5k, elites/event 10k, bosses 20k; 8 particles; one seeded random legal move in first 24 combat decisions in collection, overworld eps 0.1; evaluation greedy. Start combat ab-gen1 and overworld v3 iter003. Record canonical combat_v4 initial state + exact actions, separate search annotations/results and derived tensor cache. Record overworld_v1 public macro-state trace, choices, transitions and separate policy annotations (not full GameContext replay).

Pipeline: validate serialization/replay and integration on a small pilot, then paired initial evaluation. Collect frozen pair; train combat candidate; paired gate holding overworld fixed; collect chosen combat; train overworld candidate; paired gate holding combat fixed; repeat until deadline. Fresh confirmation reserved at end. Evaluation seeds never train. Failures stop stages; incomplete runs do not count as losses. Keep incumbents on weak evidence. Outputs managed by runs.run; scripts resumable. One CPU-heavy stage at once.

Runtime/batch sizing measured by pilot, not assumed. Hard wall budget 9 hours from controller launch; reserve 45 min for final fresh paired evaluation/report. No architecture sweep, no new objective/horizon tonight. Existing Act-1 objective may sacrifice later-game strength; no full-game claim.

Status: implementing; no training/gameplay launched. Only bounded helper task: simulator BattleContext snapshot/restore/test. All controller, schemas, writers and learning integration owned here.
