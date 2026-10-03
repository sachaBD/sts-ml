# Neural combat (PV) — session plan, 2026-10-03/04 (~16 h autonomous; user-approved)

Terminal goal: the strongest general autonomous A20 Ironclad agent (Heart kills). This session's bet: combat is the
bottleneck, so build a policy/value network + PUCT search (agents/combat/pv) that beats guided-rollout MCTS
("teacher", 20k sims, 8 particles). Milestones are means, not ends: on a dead end, redirect toward the general agent
and log why.

Stages (session target: 1 done, 2 started):
1. **Champ.** Bootstrap from teacher visits/outcomes (~1.9k recorded Champ fights) → iterate search/network design
   until paired win rate beats the teacher (planning signatures as diagnostics, not the goal) → self-play on
   generated decks until plateau (2 rounds without gain > 1 SE).
2. All three Act 2 bosses (shared trunk + per-boss conditioning/heads; ablate), then Act 2 elites.
3. Act 3 bosses + Act 2. 4. Heart + Shield & Spear. Then overworld with PV in the fights where it helps.

Evaluation: paired against the teacher on identical held-out starts (fresh seeds 981e9+, never trained on).
Primary: win rate; secondary: HP-equivalent score (0 loss; 35 + HP + 4·potions win). ± = 1 SE of paired diffs.
Cost target ≤ ~14 s per Champ fight (teacher cost), flexible.

Champ diagnostics (combat_v4_full decompressor rows + DuckDB): turn/decision of crossing 50%, player/Champ Strength
and player HP then, scaling plays before crossing, turns crossing→kill, Executes taken, debuffs spent before Anger,
search depth (actions and turns, visit-weighted, quantiles).

Decks for self-play: real recorded fight starts (Act 2+ decks, relics, potions, HP) moved to the Champ room with a
fresh battle seed; augmentation adds/removes scaling cards and jitters HP.

Storage: Parquet + DuckDB only. JSON only as process pipe transport, never stored.
Roles: me = design/decisions/training/analysis/log; impl-26-10-3 = defined code tasks; orch-26-10-3 = long runs.
Docs: LOG.md (decisions, UTC), RUNBOOK.md (launches), REPORT.md (results).
