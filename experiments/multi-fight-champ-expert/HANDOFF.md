# Handoff (written 2026-10-08 ~22:00 UTC, for the next session)

## State

- Overnight job running since 20:58 UTC: `overnight.sh` = train `champ-five-rollout-v1` (5 fights, 10 updates)
  -> test `champ-five-test-v1` (update-10 model + MCTS20k, 300 fresh seeds/deck; two-fight model on barricade and
  demon-form). Estimated ~7.5 h (rough). The user and/or an SRE agent are monitoring (`RUNBOOK.md`).
- Check first: `experiments/multi-fight-champ-expert/monitor.sh` and the log `scratch/multi-fight-champ-expert/overnight-20261008T205837Z.log`.
  On failure: do not relaunch or `--overwrite`. The app has no resume, and the test seeds come from the run id.
- Results so far are in `README.md`. Two-fight model: beats MCTS20k on both fights, matches the Barricade specialist,
  and is within noise of the Demon Form specialist (-1.8 [-4.5, +0.9], paired).

## Tomorrow

1. Read `champ-five-rollout-v1/out/REPORT.md` (monitor curve; the same 50 seeds per deck every update, so descriptive).
   Questions: does demon-form climb after update 1, and do the 3 new fights learn? Apparition was 3/20 under MCTS,
   so its bootstrap data is mostly losses.
2. Read `champ-five-test-v1/out/REPORT.md` (the result). Per deck: five-fight model minus MCTS. On barricade and
   demon-form: five-fight minus two-fight (did adding fights cost anything?). About +-4-5 pt paired intervals at 300 seeds.
3. Add results to `README.md`. Keep it concise.
4. Likely next: if curves are flat after update 1, test the update step (more epochs, or greedy collection,
   `explore = false`) as one controlled change, not more fights.

## Gotchas

- Use paired tests (`evaluate.py`, same seeds) for model comparisons. Unpaired comparisons were off by ~3 pts.
  `seed_namespace` plus `kind = "results"` adds agents to an earlier test without replaying.
- Budgets in configs are PER DECK. Monitor fights reuse seeds; never treat monitor numbers as a final result.
- `apps_garbage/run_rl_combat/` holds retired scripts that older experiment notes reference. Don't revive them.
  `apps/human_champ/bench.py` (untracked) is a dependency of the new app.
- The user wants it simple: small focused changes, no sprawl, ask before broad exploration.
- Nothing from this work is committed yet.
