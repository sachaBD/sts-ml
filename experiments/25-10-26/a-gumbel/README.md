# A. Gumbel search at low sims (C++ / CPU)

Goal: cut search to 128–256 sims with Gumbel root search and train on improved-policy targets, for far cheaper games.
Paths considered: [../possible-paths.md](../possible-paths.md).

## Steps 1–3 (25 Oct 2026): eval-only gate, no training

- **Build:** `build/a-gumbel` (pv_worker + root-halving tests pass; ~3 min). Old frozen worker no longer loads
  (repo path moved). Parity with frozen `58c5c4ab` assumed, not replayed (user decision). Log: `logs/build.log`.
- **Fix:** `apps/megacrit_dump/champ_starts.py` still pointed at `../sts_lightspeed`; now `../lightspeed`.
- **Gate:** `gate.py`. Corpus-pilot round-4 model, same 400 dev starts (20 unseen families × 20 seeds), greedy,
  10 workers. Gumbel = existing root-only Sequential Halving, m=16, gumbel_scale 0, interior PUCT.
  PUCT2k and MCTS20k arms are the stored pilot results (not replayed). Results:
  `runs/schema=combat_v4/date=2026-10-25/id=a-gumbel-gate-v1/out/report.json`. Log: `logs/gate.log`.

| arm | wins/400 | vs PUCT2k, pp [95% CI] | s/game | ms/decision (median) |
|---|---:|---:|---:|---:|
| MCTS20k (stored) | 201 | | 15.5 | 367 |
| PUCT 2000, rollout .5 (stored) | 183 | — | 9.1 | 233 |
| **Gumbel 256, rollout .5** | **153** | **−7.5 [−11.8, −3.3]** | 1.4 | 32 |
| PUCT 256, rollout .5 (control) | 129 | −13.5 [−21.0, −6.8] | 1.3 | 31 |
| Gumbel 128, rollout .5 | 116 | −16.8 [−23.3, −10.8] | 0.7 | 14 |
| Gumbel 256, no rollout | 100 | −20.8 [−31.5, −11.5] | 1.3 | 28 |
| Gumbel 128, no rollout | 75 | −27.0 [−37.3, −17.5] | 0.6 | 12 |

CIs: equal-weight family mean paired difference, family-cluster bootstrap (as the pilot). 0 caps/errors.
Gumbel256 − PUCT256 (same sims): +6.0 pp [−0.3, +13.0]. s/game includes worker start-up per fight.

Findings:
- 256 sims costs ~7× less per decision than 2k. The network dominates; rollouts add only ~10–15% (10-game timing, `timing/`).
- At equal sims Gumbel beats PUCT (+6 pp, CI just touching 0), but Gumbel256 is still −7.5 pp below PUCT2k.
  The planned gate ("≈ PUCT2k at ≥5× cheaper") **fails on play strength**.
- Rollouts matter a lot at low sims (−13 to −11 pp without them): the value net alone is the weak link, not rollout noise.
- Caveat: this model was trained on PUCT-2k visit targets; it measures play strength, not Gumbel training-target quality.
  Different worker build/simulator commit than the stored arms (assumed equivalent).
