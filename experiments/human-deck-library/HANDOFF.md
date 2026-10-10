# Incoming overnight orchestrator — handoff

## Coordination and ownership

User says **do not initiate any messages**. The incoming agent will contact this session.
Do not start duplicate jobs or compete for the same output lock. Current owner has NOT
stopped the library run. User permits stopping this specific run if useful, not other jobs.
All changes are uncommitted; preserve unrelated edits. Future gameplay concurrency: 10.
Always provide a copy-paste overall dashboard command, not only the current stage tail.
Announce long-run purpose/runtime uncertainty/log path; background it with preserved logs.

## Proven result / anchor task A

`experiments/single-deck-expert-iteration/FINAL_RESULTS.md` documents independent final
confirmation: learned2k 595/600 (99.17%, 95% Wilson 98.06–99.64%), MCTS20k 514/600
(85.67%). Paired gap +13.50 ±1.42 points (1 SE), approximate 95% +10.72–16.28 points.
82 learned-only wins, 1 MCTS-only win, 4 both lost, 513 both won. All plays completed.
Fresh width64 model (no D5 initialization), bounded sigmoid value head; 200 teacher
bootstrap + 500 learner fights. Value labels actual eventual win ×100; visit policy
labels with concentration weighting. No HP reward. Search stays; no architecture overhaul.
Teacher replay at update5 =120 fights; learner replay=500; not self-only learning yet.
Fixed A: `7a9ada48-a4d6-41fc-a85b-0147618df00a`, Champ A20, 41/75 HP, no potions.
Model: `runs/schema=combat_v4/date=2026-10-05/id=single-deck-fresh-v1/out/iter005/model/`.
Frozen worker: `runs/schema=combat_v4/date=2026-10-05/id=single-deck-fresh-final/out/frozen/pv_worker`.
Do not overwrite these results/models. Final 600 seeds are now consumed, not untouched.

## Live library run

`runs/schema=combat_v4/date=2026-10-05/id=human-deck-library-v1/`.
Runner: `apps/human_champ/library.py`, reusing `apps.human_champ.bench` play and
`apps.run_rl.combat_loop` stage orchestration. Plan: `experiments/human-deck-library/PLAN.md`.
20 loadouts ×20 MCTS20k seeds: 40 cached reference trials +360 new trials, 10 workers.
18 additional selections are deterministic diverse card families, stratified by engine.
Original HP varies per loadout, relics/counters fixed, no potions. No outcome-based dropping.
Raw `out/starts.parquet`, `new-starts.parquet`, `manifest.json`, `reused-results.jsonl`.
Final catalogue is only written after play finishes. Dashboard reconstructs live counts.

At handoff inspection: 60/400 persisted results (two cached refs plus 20 new results on
`51a5aff9...`, 13/20 wins). Ten native workers actively ~100% CPU each, some running >4 min.
No traceback in root stderr. **Not proven hung.** Initial 8–12 minute forecast was too
optimistic for larger decks, and ordered result collection makes it look stalled.

`apps/human_champ/bench.py:215` uses `pool.map(play_one,jobs)` and journals submission-order
results. Head-of-line blocking hides later completed jobs. Native jobs have per-fight
600-second subprocess timeouts. A restart resumes persisted journal IDs, but will lose
finished-yet-unpersisted results buffered behind a slow job. Consider improving to
completion-order collection (`submit`/`as_completed`), verifying ID joins/summary/replay
ordering, then performing a narrowly scoped stop/restart if warranted. Do not edit/replace
the frozen native worker in place. Do not classify timeouts/errors as losses or quietly
report partial completion as full evidence. Watch timeout/resource cost and set a wall-time
budget before committing to an overnight schedule.

Inspection process IDs (stale quickly; re-resolve before signaling): benchmark controller
174230, parent library controller174168; native children have argv under this run's
`out/frozen/pv_worker teacher 20000`. Terminate only verified descendants of this run if
needed. bg job10 in original session; launcher log `/tmp/human-deck-library-v1-launch.log`.
Stage log `logs/stdout.log`; native progress `out/logs/play.log`; persisted outcomes
`out/play/results.jsonl`. The resumable managed command is recorded in `run.json`.

Dashboard:

```sh
watch -n 5 '.venv/bin/python -m apps.human_champ.library status --run runs/schema=combat_v4/date=2026-10-05/id=human-deck-library-v1'
```

## Proposed next experiment — discuss/confirm execution bounds with user

User wants overnight orchestration and joint analysis tomorrow. Proposed research idea:
**add one new fight at a time, train jointly with retained old-task replay**, not isolated
specialists that forget each previous deck. Existing agent recommendation: initialize from
A specialist; new task B gets ~75% training sampling, A gets25% replay anchor. Generate new
experience mainly on B; regenerate A only if its policy regresses. Track each separately.

Queued B candidate: user's shorthand `002005aa1` (original `00505aa1`); actual matching
stored deck `00205aa1-6004-4ab0-8600-655bddec017e`, 16 cards including Flex+/JAX+/Limit
Break+/Heavy Blade+, HP33/80. MCTS20k selection screen18/20 (95% Wilson69.9–97.2%), all
completed, source `runs/schema=combat_v4/date=2026-10-05/id=strength-deck-mcts20/`.
No current-specialist transfer evaluation on B has run. That small paired check is the
cheapest immediate next question. Library could provide alternatives: hard for learner
but demonstrably recoverable by teacher, not merely intrinsically doomed low-HP starts.

Suggested stages, not a claim they are launched/implemented:
1. Finish bounded library and expose failures/time costs.
2. Evaluate current A model on selected library starts to identify transfer gaps, budget
   this explicitly (all20×20 is400 learned plays; expensive cases may need a smaller screen).
3. Choose one contrasting, affordable B task. Freeze new train/monitor/final namespaces.
4. Teacher bootstrap B, then learner-search self-play with A replay anchoring. Preserve
   optimizer moments when resuming, equal states per fight, actual outcome labels and
   concentration policy weighting. Existing learner supports explicit fight inclusion,
   optimizer resume and bounded head. Do not feed monitoring/final trajectories to training.
5. Plot A/B separately; retain original model as rollback. No claim of general superiority
   from two decks. Do not silently expand deck count/epochs/search budget overnight.
6. Morning report: seeds/sample sizes, CIs/gaps, statuses/timeouts, wall times, resources,
   exact model selection, forgetting/transfer curves, what remains untested.

Current `single_deck.py` is hardcoded for A, so joint A/B requires a small explicit extension
or a dedicated orchestrator reusing existing PV train/data and combat_loop stages. Do not
pretend the current one-deck controller already implements 75/25 multi-task replay.
Existing monitoring100 A seeds are available for regression; previously final600 cannot
remain an untouched test once used in future tuning. Allocate new final seeds for any
new adaptive recipe and keep discovery/library seeds labeled as selection data.

## Verification / caveats

Focused earlier20 unit tests passed; latest library/dashboard7 tests pass. Modules compile.
Library selectors dry-run20 unique families:4 block,3 Demon Form,4 exhaust,5 strength,4 mixed
(including references). Reference worker checksums match current frozen worker, checked
before launch. Historical reconstruction/balance caveats are unchanged. Rollout teacher
also values HP/resources while learned recipe targets wins; no ablation established the
source of the gain. Larger decks' MCTS CPU variability is currently unknown.
