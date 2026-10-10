# Handoff — human-deck corpus (2026-10-06, fight-expert-26-10-5 → user)

Status: **stopped. Nothing running.** The machine restarted at ~04:38Z during eval-20. No S3 confirmation was played,
so every reserved final seed is untouched. `control.json` has `stop: true`.

## Bottom line
- The single-deck +13% recipe **reproduced on A** in the joint run (30/30 monitor by update 4, MCTS 28).
- It did **not** generalize as a corpus builder:
  - **Easy decks (MCTS ≥80%):** learned is near MCTS, but none clearly exceed it, with ~200–480 learner fights each.
  - **Hard decks (MCTS ≤60%):** flat to slightly declining, well below MCTS, despite 525–700 learner fights each
    (more than A ever got).
- At k=16 (30 monitor seeds/deck, selection-consumed): 6/22 decks were at or above MCTS. Only A and 3cbcfe8a are
  above it by more than noise. 14/22 improved vs their first eval; 6/22 got worse.

## My mistakes (please weigh the results with these in mind)
1. **Graduation bar never adapted.** "≥27/30 and ≥ MCTS" made "graduated" meaningless for decks whose best case is
   below 90%. Reporting "2/23 graduated" obscured the real picture. It should have been per-deck relative to MCTS
   (e.g. ≥ MCTS with a paired test).
2. **I kept expanding on a null result instead of investigating.** By k=7 the pilot's hard decks had plateaued
   (collection wins flat at 72–89/350 for six updates). I made two bundled changes (teacher taper only after
   mastery, easy-first waves) and kept adding waves. I should have stopped and run controlled diagnostics on the
   hard-deck plateau first, e.g.:
   - more teacher data vs none;
   - exploration on/off for labels;
   - fights/update matched to A;
   - single hard deck at A's budget;
   - HP raised to make it easy.
   The two S2 changes were never isolated, so their effect is unknown.
3. **Separate-vs-joint was over-read.** n=1 deck, and joint had ~8× more total gradient data. Later, 168ec20d fell
   18→12→11 as waves were added, which is consistent with dilution/interference. Treat "joint > separate" as weak.
4. **Ops:**
   - ~45 min lost to an encoder-unsupported card crash (RAGNAROK) while I was in a long sleep.
   - RUNBOOK timestamps between ~02:36Z and 03:33Z are mislabelled by +1 h; the controller logs local BST. A
     correction note is in RUNBOOK.

## Evidence worth keeping
- **Win-only diagnostic (single-fight-investigation):** a win-only rollout MCTS rescued 2/44 learned-only seeds and
  regressed both sanity wins. The HP objective does not explain A's +13%.
- **Hard-deck value check:** first-decision value V0 vs actual win%. V0 is roughly calibrated to the learner's own
  play, slightly optimistic. There's no "give up" collapse in aggregate, so the stall looks like policy improvement
  failing below MCTS. Descriptive only; not a controlled test.

  | k | hard-7 V0 | hard-7 win% |
  |---|---|---|
  | 0 | 52 | 23 |
  | 4 | 23 | 26 |
  | 8 | 25 | 31 |
  | 12 | 30 | 29 |
  | 16 | 37 | 27 |

  MCTS wins 41% on the same decks.
- **Zero-shot on entry (wave 1, k=8 model):** 104/210 (50%) vs MCTS 180/210 (86%).
- **Per-deck table at k=16:** a NOTES.md-style table is reproducible from `state.json`. REPORT.md and curves.csv are
  in the run dir.

## Untested ideas I'd prioritize
1. One easy deck and one hard deck, each at exactly A's budget (200 teacher + 5×100 learner), to separate
   "undertrained/diluted" from "recipe fails at low win rate".
2. HP curriculum for hard decks: raise HP until MCTS is ~85–90%, master that, then step HP down.
3. Keep the teacher in the loop for hard decks, e.g. MCTS labelling learner-visited states, instead of tapering it.
4. A relative-to-MCTS graduation rule with a paired test, and per-deck budgets matched to A, before any further
   breadth expansion.

## State / artifacts
- **Run (pilot + S2, same dir):** `runs/schema=combat_v4/date=2026-10-05/id=hdc-pilot-joint/out/`
  - `state.json`, `REPORT.md`, `curves.csv`, `logs/controller.log` (local time), `logs/s2-launch.log`.
  - Models are in `models/<k>/model/`; k=20 is the latest trained.
  - Backups: `state.pilot-k8.json`, `state.pre-fix-k9.json`, `config.pilot.json`.
  - eval-20 is partial (465 of ~690 results). Rerunning `experiments/human-deck-corpus/s2-command.sh` resumes it,
    then exits because `stop: true` is set.
- **Control:** `runs/schema=combat_v4/date=2026-10-06/id=hdc-pilot-control-168ec20d/out/`.
- **Splits:** `runs/schema=combat_v4/date=2026-10-05/id=human-deck-corpus/splits/`.
  - 1,028 pool decks; 540 held-out families (never screened or trained); 8 consumed controls.
  - 3 library decks are training-protected.
- **Screen:** `runs/schema=combat_v4/date=2026-10-05/id=hdc-screen-w1/` (40 decks × 20 MCTS20k).
- **Deck lists:** `runs/schema=combat_v4/date=2026-10-05/id=human-deck-corpus/decks-v1/`
  - `decks-v3-easyfirst-encodable.json` is the current file; `skipped.json` lists exclusions.
  - Waves 3–5 (21 decks) never entered.
  - Excluded:
    - 6fe8f9ef: RAGNAROK, encoder-unsupported.
    - abe6e6de: ALPHA, encoder-unsupported.
    - cf0e8f0b: capped fights, likely learned stalling, marked software-failed by rule.
- **Final seeds:** `final-reserved.parquet` in the run dir was regenerated at the last entry; 30/deck, never played.
  `apps/corpus/confirm.py` (S3) is written but **not yet run on real data**.
- **Win-only diagnostic:** `runs/schema=combat_v4/date=2026-10-05/id=barricade-win-only-diagnostic/`.

## Code (all uncommitted)
- `apps/corpus/{splits,screen,decks,controller,confirm}.py`, `apps/corpus/test_controller.py`.
- `agents/combat/pv/{data,train}.py`: opt-in `--mix-shards`, `--groups`, and a duplicate-fight error.
- `agents/combat/pv/test_mixed_shards.py`.
- 10 unit tests pass.
- impl's hardening changes are in the code: entry encode failure → deck marked software-failed, and
  `decks.py --check-encodable`. They are untested in a live run.
- `controller.py` has `taper_after_mastery` and allows the decks file to change on resume.

## Helpers
orch-26-10-5 and impl-26-10-5 have no pending tasks. single-fight-investigation was used only for the diagnostic
and the plan review.
