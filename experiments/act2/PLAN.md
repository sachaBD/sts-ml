# Act 2 extension — 2026-10-02 (approved by user ~08:00 UTC, ~10 h autonomous)

Terminal goal: strongest autonomous full-game agent. This session: broad progress — a working Act 1+2 agent with
Act 1+2 data and an overworld value net trained on both acts. Not tuning Act 1 (don't regress it).

Decisions (user-agreed):
- Combat = guided-rollout MCTS (no neural leaf). Budgets easy/hard/elite/event/boss 500/5k/10k/10k/20k, 8 particles.
- Target: linear in floor reached + step bonus for Act 1 clear + bigger step for Act 2 clear. Watch for short-term bias.
- Exploration on every overworld decision; some runs follow random routes (to reach elites).
- Boss relic choice via overworld value (after-states).
- Issue #5: event card rewards into macro decisions.
- Run ends after Act 2 boss. Simulator gaps -> SimpleAgent fallback + recorded, no simulator rewrite.
- Out of scope: Act 1 tuning, Hexaghost, neural combat, arch sweeps, Act 3.

Roles: me = design/decisions/log + Python (target, policy, exploration, training); impl-10-2 = worker C++ (act 2,
boss relic, event cards); orch-10-2 = launching/monitoring runs via runs.run, one CPU-heavy stage at a time.

Phases: engineering (~2 h) -> smoke (~50 runs) -> collection (~2 h) + train -> 2-3 gated rounds -> fresh final + report.
Token constraint: jobs ~40 min or ~2 h; work in bursts.
