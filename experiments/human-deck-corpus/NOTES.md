# Notes / decisions

## Context inherited (2026-10-05)
- Single-deck A (Barricade 41/75): learned2k 595/600 vs MCTS20k 514/600, +13.5 ±1.4 pp (1 SE). 82/1 discordant.
- General EXIT attempts plateaued: combat-pv best 30.8% vs teacher 41.8%; champ-oracle-exit ~39–41% vs 41.8%;
  D5 on 431 human decks −10.3 ±1.6 pp vs MCTS.
- Library v1 (20 loadouts × 20 MCTS20k): wide spread 0/20 → 20/20; MCTS mean fight time 4–171 s/deck.
  Scrawl unsupported → deck excluded.
- Earlier claim "much of the 30 min was overhead" was wrong: ~19 of 30 min was gameplay.

## Win-only objective diagnostic (owned by single-fight-investigation)
Fixed-scale win-only MCTS20k (W=(35+maxHP)/(55+maxHP), loss/unresolved 0) on 44 of the 82 learned-only seeds,
4 reproduction gates, 2 both-win sanity. Rescue diagnostic, not attribution.
**Result (22:43):** all 4 gates reproduced exactly; win-only rescued **2/44 (4.5%, 95% Wilson 1.3–15.1%)**; both
sanity wins became losses; 50/50 completed. → Objective mismatch does not explain the learned agent's edge; a
naive win-only rollout teacher is worse, not better. Do NOT use a win-only teacher for bootstrap.
Residual is not proven to be learned evaluation (priors/tree/UCB calibration differ).
Artifacts: runs/schema=combat_v4/date=2026-10-05/id=barricade-win-only-diagnostic/out/.

## Open decisions
- Pilot deck choice: rule-based (archetype spread, MCTS 20–90%) unless S0 surprises.

## Pre-registered: S1 separate-specialist control (decided 22:20Z, before any pilot results)
Deck **168ec20d** (demon_form, 62/80 HP, MCTS 11/20, fastest mid-rate non-A pilot deck). Same controller, decks.json
with only this deck (wave 0), same namespace hdc-v1 → identical teacher/monitor/learner seeds and fight IDs, same
exclude set, fresh init (torch seed 0), bootstrap 10 ep lr1e-3, then 8 updates × 50 fights, same control values.
Experience-matched (same per-deck fights), not compute-matched. Runs after the joint pilot (never concurrently).
Read-out: monitor wins/30 for 168ec20d at k=0,4,8 joint vs separate; plus A joint vs A's single-deck curve
(note A single-deck used 200 teacher + 100/update — different recipe, descriptive only).
Pilot decision rule (S1→S2): joint ≈/≥ control on 168ec20d (within ~±4/30) and ≥5/8 decks with monitor rising from
k=0 to k=8 → scale joint with curriculum. Otherwise: teacher coverage 100/deck & inspect before any width change.

## S1 pilot result (joint, 8 decks, k=0→8; monitor wins/30 vs MCTS20k ref)
A: 26→30→30 (MCTS 28) graduated at k=4. Non-A sum: 48→55→65 of ~210 (23%→31%) vs MCTS 87/209 (42%).
At/above MCTS by k=8: 168ec20d 18 vs 16, 327e6f93 12 vs 9. Collection wins (explore) non-A: 34/350 at k=1, then
flat 72–89/350 (k=2–7): plateau after first update. Train RSS grows ~0.3 GB/update (4.0 GB at 2.5k fights).

## S2 adaptation (00:10Z) — two changes, both logged in control.json / code
1. `taper_after_mastery`: keep the full 50 MCTS teacher fights per deck until the learner first matches MCTS on that
   deck's monitor seeds; taper 5/update after that. Rationale: pilot tapered teacher to 10/deck by k=8 while the
   learner was still well below MCTS on hard decks (taper was designed for A where learner already > teacher).
   mastered_at set retroactively from pilot eval history (A k4, 168ec k8, 327e k8, 3cbc k0).
2. Waves easy→hard (decks-v2-easyfirst.json: MCTS 19/20 → 1/20, 20/20 anchors last) — grow the corpus where
   the recipe is known to work (A: MCTS 90%), then hard decks benefit from a stronger shared net.
Other S2 settings: curriculum ON (min 10/deck, 480/update), wave every 3 updates, eval every 4, replay 200/deck
(RAM bound), continuing the pilot model and state in place (same out dir, controller run directly, not runs.run).
Not changed: exploration (root noise + early-turn sampling), width, losses. Candidate next lever if hard decks stay
flat: exploration lowering value labels (collection p ≈ half of greedy monitor rate).

## Hard-deck plateau diagnostic (03:45Z, existing eval data, no new compute)
First-decision root value V0 (learned2k, monitor seeds) vs actual win%:
hard-7 decks k=0/4/8/12/16: V0 52/23/25/30/37 vs win 23/26/31/29/27% (MCTS 41%); easy wave1 k=12/16: V0 64/72 vs 68/74%.
→ No value collapse toward 0 ("giving up") in aggregate: V0 is roughly calibrated to the learner's own play (slightly
optimistic on hard decks). Plateau looks like policy improvement stalling below MCTS on hard decks, not pessimism.
Descriptive; not a controlled test.
