# Retired combat scripts (from apps/run_rl), pending deletion

Moved here on 2026-10-08. They had been added to `apps/run_rl` (the overworld app) for the single-deck and
expert-champ experiments. They are kept so the records in `experiments/single-deck-expert-iteration/`,
`experiments/expert-champ/` and `experiments/combat-agent-rescue/` stay traceable. Their imports
(`apps.run_rl.*`) no longer resolve, and they are outside the test glob.

Replacement: `apps/combat_expert_iteration/`. The two dashboards are standalone and still run by path, e.g.
`.venv/bin/python apps_garbage/run_rl_combat/expert_champ_status.py --run RUN_DIR`.
