# Fixed-scale win-only rescue diagnostic — approved, in progress

Question: can objective alignment rescue seeds where learned2k won and legacy MCTS20k
lost? This is an enriched counterfactual screen, **not causal attribution of the net13.5pp**.

Protocol agreed with fight-expert-26-10-5 and approved by user:
- Fixed win-only value W=(35+root max HP)/(55+root max HP)=110/130 on this deck.
  Loss/unresolved0. Remove terminal HP/resources/turn/loss-progress shaping. UCB0.7,
  old normalizer, guided rollout,8 particles,20k simulations,512 maximum actions unchanged.
- Uniform sample without replacement44 of the82 learned-only final cases (Python
  Random seed20261005); two both-win controls. IDs frozen before build/outcomes.
- Four rebuilt-legacy exact-action/outcome/HP gates (two losses +two wins).
- If gates pass:44 win-only rescue games +two win-only controls,50 total including gates.
- Approved fallback if gates fail:23 cases ×two rebuilt objective arms plus the four
  gate games,50 total. This is not a comparison to a bit-identical cached baseline.
- Hard dispatch cap50, no automatic game retries, no training or checkpoint selection.
- Build/selftest failures halt; diagnose within bounds, no silent protocol/budget expansion.
-10 gameplay workers maximum, wait for existing library CPUs. Build uses two compiler jobs.

Implementation: `apps/human_champ/win_only_diagnostic.py`. It snapshots minimal teacher
worker code from current apps/pv/worker.cpp, plus a copy of the simulator's PublicBeliefCombatSearch
translation unit. Only copied implementation gets opt-in mode2. Existing int field/API
means no header/ABI change. The dedicated executable links combat_core, with its copied
PublicBelief methods supplied by the executable rather than the archive's default object.
No shared simulator or working-tree worker edit, no overwritten frozen binary.
Isolated build `build/win-only-diagnostic/`; generated source copies also preserved in
managed `out/source/`. Runtime selftest covers low/high HP and resource invariance,
loss/unresolved/escape/zero-HP handling without consuming any games. Reproduction gates
are essential because current sources and compiler configuration may differ from frozen build.

Run: `runs/schema=combat_v4/date=2026-10-05/id=barricade-win-only-diagnostic/`.
Selection/gates: `out/config.json`, `gates.json`; dispatched IDs: `out/dispatched.json`;
completion-order raw outcomes: `out/results.jsonl`; eventual results: `out/summary.json`.
Build log `out/build.log`; stage log `logs/stdout.log`; errors `logs/stderr.log`.

Dashboard:

```sh
watch -n 5 '.venv/bin/python -m apps.human_champ.win_only_diagnostic status --run runs/schema=combat_v4/date=2026-10-05/id=barricade-win-only-diagnostic'
```

Report rescue counts and uncertainty, sanity outcomes, reproducibility, statuses and artifact
path. Distinguish unfinished whole games from internal unresolved rollouts: only the former
is measured in this diagnostic. Remaining failures do not uniquely imply learned evaluation;
learned priors, PUCT/tree implementation, sparse rollout rewards and calibration differ.
Five/two positive controls cannot estimate regressions among all original teacher-winning seeds.
