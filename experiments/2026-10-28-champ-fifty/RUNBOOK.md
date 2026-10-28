# Champ fifty-deck expansion — 2026-10-28

## Goal and guardrails
Expand the original ten Champ decks to fifty, warm-starting the working Gumbel agent.
Keep architecture fixed (64/2/2), 5,000 total games/update, five-update replay,
deck-balanced training, minibatch-only GPU loading, approximately 20% validation wall time.
Keep the original ten as a separate retention cohort. Report paired 95% improvements
against the starting checkpoint; historical MCTS20k/PUCT comparisons stay explicitly historical.

The existing ten-deck run must continue throughout preparation. Stop it only when the
replacement is ready. On repeated preparation failure abandon the expansion; on
post-switchover failure restore the known-working run rather than waste the day.

## Working run / rollback anchor
`runs/schema=gumbel_nn_combat_v1/date=2026-10-27/id=gumbel-nn-champ-ten-20261027-205535`
Current session log: `logs/resume.log`; dashboard uses actual loop lock, not stale run.json.
Stop: `./apps/gumbel_nn_combat_expert_iteration/run.sh stop RUN_DIRECTORY`.
Latest completed checkpoint and retained replay must be preserved before switching.

## Progress
- Preparation started. Initial working-tree snapshot will be committed before implementation.
- No replacement training job launched; existing job not stopped.

## Recovery
Detailed verified launch/recovery commands and validation results will be added here
before switchover. Never append partial gameplay. Preserve interrupted stages and
inspect native worker processes before resuming. No silent simulator fixes/reversions,
no filtering failures into losses, no automatic endless restarts.
