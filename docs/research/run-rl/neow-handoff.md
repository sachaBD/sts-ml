# Neow: handoff (2026-10-01)

## Decision (user, 2026-10-01)
**Neow is a regular decision scored by the value network V, not a bandit.** The bandit premise ("the moment is fixed")
is wrong: the map and the act boss are known at Neow and differ per run, so the choice is contextual. The bandit
approach is **retired** (`docs/research/run-rl/neow.md` and `agents/overworld/neow.py` carry RETIRED headers).

## State of the code
- Commit 0becd5e (another session): worker `decide: neow` sends the 4 arms `<bonus>|<drawback>` with NO after-states;
  `play.py --neow`, `loop.py` per-iteration `iterNNN/neow.json` route the choice to `NeowPolicy`.
- Uncommitted (this session): `agents/overworld/neow.py` has a working Thompson-sampling `NeowPolicy` (Beta per arm,
  discount 0.9, greedy eval, seeded) plus the RETIRED header; `play.py` one line (`NeowPolicy.load(neow, seed=seed)`).
  Offline-checked only (12 logged runs; synthetic bandit). Not used further. Can be committed as the retired record
  or reverted to the stub; either is fine.
- `scratch/run_v4_neow.sh`: bandit loop launcher, **never started; obsolete, do not run**.
- No real runs were made for the bandit.

## Proposed approach: Neow by V over sampled outcomes
- Problem: arm outcomes are random (which rare card, boss relic, transform, potions), so an exact after-state would
  peek at the real roll.
- The worker already has the needed machinery for events (`decide: event`, see `apps/run_rl/worker.cpp` header):
  SAMPLED LOOKAHEAD, `lookahead_samples` game copies with fresh randomness (never the real rolls; same randomness per
  sample index across options), option applied, played on with the current policy to `lookahead_horizon`, ends sent
  to Python as `{"type": "evaluate", ...}` and scored by V. Neow should reuse this: per arm, K sampled outcomes
  (follow-up screens such as remove / upgrade / choose-a-card answered by the current policy inside the sample),
  score = mean V of the resulting floor-0 states (horizon 0 = right after Neow resolves).
- The after-state already includes map + boss, so the context the bandit ignored is used automatically.
- Then: `--decide ... neow` in the loop; paired fresh-seed comparison vs the same model without neow.

## Risks / open questions
- **V has almost no data on these states.** SimpleAgent always took slot 1, so boss-relic swaps, curses, no gold,
  250 gold / -30% HP etc. are unseen at floor 0. V will extrapolate at first; exploration (eps) on the Neow choice is
  needed so the loop collects them. Watch for a systematic preference (cf. the early "remove Bash" artefact in v3).
- Does V's encoding represent everything Neow changes (relics, potions, curses, max HP, gold)? Check before trusting.
- Act-1-only objective: boss relic / gold judged by Act 1 clears only.
- Cost: K samples x 4 arms per run, once per run: cheap.

## Data: dedicated Neow table (agreed in principle, not built)
One row per run, `runs/schema=run_rl_v1/.../out/neow.parquet`, view `run_rl_neow`, joins `run_rl_results` 1:1.
Columns are neutral to the decider so the table outlives the bandit:

```
part VARCHAR, iter INTEGER, seed BIGINT, boss VARCHAR
arm0..arm3 VARCHAR          offered arm per slot '<bonus>|<drawback>'
choice INTEGER, arm VARCHAR chosen slot / arm
simple INTEGER              SimpleAgent's index (always 0)
source VARCHAR              net | explore | simple
value0..value3 DOUBLE       what the decider compared (mean V over samples); NULL if simple
policy_ref VARCHAR          model checkpoint used, relative to the run's out/
result VARCHAR              what the arm actually gave (e.g. 'boss relic: sozu'); needs the worker to report it
```
Reproducibility: with fixed sample randomness per index (as the event lookahead does), the choice is reproducible
from `policy_ref` + seed + K; export checks `choice` = argmax(values) on every row.
A general `run_rl_decisions` table (path / rest / shop / card / event, with floor, hp, max_hp, gold, deck_size,
options, choice, simple, source, values, policy_ref) was also discussed; Neow stays out of it.

## Next steps
1. Extend the worker's Neow decision to the sampled-lookahead path (reuse the event code), report `result`.
2. `play.py` / `loop.py`: route Neow like `event` (V over sampled ends, eps exploration); drop the `--neow` bandit file.
3. Export the Neow table; short real loop; paired fresh-seed comparison. Coordinate compute with the user (11 cores).
