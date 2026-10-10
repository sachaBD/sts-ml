# Spec: multi-deck corpus controller (`apps/corpus/controller.py`) + mixed-batch training

Owner: fight-expert-26-10-5. Implementer: impl-26-10-5. Read PLAN.md first.
Reuse; don't rewrite: `apps.run_rl.combat_loop.{play,encode,stage,read_results,write_starts,sha}`,
`apps.run_rl.single_deck.{wilson}`, `agents.combat.pv.train` (one small extension below),
`agents.combat.pv.data` (unchanged encoder). Model: `PolicyValue` width64, `--value-activation sigmoid` when fresh.
Keep it simple and readable; single file controller (~400 lines max) + status command + tests.

## Inputs
- `--decks decks.json`: list of `{deck_id, start(row dict from pool.parquet), teacher_runs:[results dirs], teacher_fight_ids:[...]}`
  (screen fights; existing combat_v4 fights/search parquet already exist in those play dirs → encode them once).
  Also `wave` (int) per deck: wave 0 = initial set; later waves enter on schedule.
- `--worker` (frozen copy into out/frozen, checksum checked like single_deck), `--out`, `--workers` (1–10, default 10),
  `--namespace` (default `hdc-v1`), `--init` optional model dir (else fresh bootstrap), `--device cuda`.
- `out/control.json` (re-read at the start of every update, written with defaults if absent):
  `{"fights_per_update": 400, "min_fights_per_deck": 10, "wave_every": 6, "wave_size": 8, "eval_every": 4,
    "replay_fights_per_deck": 300, "teacher_per_deck": 50, "epochs": 3, "lr": 3e-4, "stop": false}`.
  `stop: true` → finish current stage cleanly and exit 0. Log every control change in the controller log.

## Seeds (deterministic per deck, independent of other decks)
`seed = sha256(f'{ns}:{deck}:{split}:{i}')[:8] little-endian`, skipping a STATIC set of excluded seeds
(all pool/bench/library/screen seeds), never mutated by other decks (so a single-deck control run with the same
namespace gets identical starts/IDs). fight_id `f'{ns}:{deck}:{split}:{i}'`. Splits per deck:
`teacher` (top-up to teacher_per_deck minus screen fights), `monitor` (30), `final` (30, written to
final-reserved.parquet, NEVER played by the controller), `learner-{k}` (k = global update).

## Per-deck entry (when its wave enters; wave 0 at start)
1. One teacher play stage (MCTS20k, teacher=True) for all entering decks: teacher top-up + monitor reference.
   Encode. Monitor reference rows = validation rows (split manifest `val`), teacher top-up + screen = `train`.
2. If a model exists: **zero-shot eval** = learned2k on that deck's monitor seeds (no exploration), before any
   training on the deck. Record per deck.
3. Wave 0 with no `--init`: bootstrap train (fresh, sigmoid, 10 epochs, lr 1e-3) on all wave-0 teacher fights.
   Then eval all wave-0 decks on monitor (that's update 0).

## Update k (global)
1. Read control.json. Enter the next wave if `k % wave_every == 0` and candidates remain.
2. Collection quotas over decks with status active/graduated: each gets `min_fights_per_deck`
   (graduated: exactly that); the remaining budget over **active** decks ∝ `max(0.05, p(1-p)) / mean_fight_seconds`,
   p = Beta(1,1)-smoothed win rate over that deck's last 2 collection batches (zero-shot batch counts),
   mean_fight_seconds from its recent learner fights (screen MCTS seconds/5 before any). Integer rounding by largest
   remainder; log the allocation table (deck, p, seconds, weight, quota).
3. Play collection (explore=True, sample-turns, 2k sims) in ONE play stage; encode → one shard per update.
   Incomplete fights: excluded from labels, counted per deck. Deck with >5% incomplete in an update →
   status `software-failed` (not parked), excluded thereafter; the run continues.
4. Training fight set per deck: last `replay_fights_per_deck` completed learner fights of that deck (by update order,
   independent of global age) + teacher fights tapered by PER-DECK age a (updates since entry):
   `n_teacher = max(0, teacher_per_deck - 5*a)`, deterministic subset. Shards passed to train.py = only shards
   containing ≥1 selected fight (prune others). Fresh decks just entered also contribute their teacher fights.
5. Train: `--init prev/model.pt --resume-optimizer --states-per-fight 64 --flat-policy-weighting --grad-clip 1
   --epochs E --lr LR --mix-shards --groups groups.json --split-manifest split.json --train-fights ids.json`
   plus validation shards (all monitor-reference rows of entered decks). Wrap with `/usr/bin/time -v` and record
   max RSS in the update's record. GPU device.
6. If `k % eval_every == 0`: learned2k on monitor seeds of all entered decks (not software-failed), paired
   against the cached MCTS reference per deck. Record per-deck wins/30, MCTS wins/30, paired gap.
7. Lifecycle (after evals): **graduated** (provisional) if latest monitor wins ≥ MCTS ref wins and ≥ 27/30;
   back to active if a later eval drops below either. **parked** if ≥150 learner fights and smoothed p < 0.05
   (stop collecting; keep replay). Parked/graduated are reversible statuses, logged with reasons.
8. Persist `state.json` atomically after every stage (update index, deck statuses, entry update, collected fight IDs per
   deck with results, eval history, allocation history, model path). Resume = rerun same command; stage markers
   (`combat_loop.stage` `.done`) make completed stages skip. Config change on resume (other than control.json) → error.

## train.py extension (minimal, opt-in; default behaviour byte-identical)
- `--groups PATH`: JSON fight_id → group (deck). Logs per-epoch realized train states per group (in the epoch JSON).
- `--mix-shards` (requires non-stream): sample states per fight as today, but pool sampled (shard,row) pairs across
  ALL shards, shuffle globally, and form each minibatch across shards (call `Shard.batch` per shard subgroup, pad
  to common counts, concat). Must give genuinely mixed-deck minibatches. Add a unit test proving a batch can
  contain rows from ≥2 shards and that per-fight sample counts equal states_per_fight.
- Fix the dup-fight trap: if a fight_id appears in more than one included shard → error.

## Outputs / status
- `out/state.json`, `out/curves.csv` (update, deck, status, collection p, monitor wins, mcts wins), `out/REPORT.md`
  regenerated after every eval: per-deck table + aggregate (macro mean learned vs MCTS; paired gap with SE clustered
  by deck) + status denominators (entered/active/graduated/parked/software-failed).
- `python -m apps.corpus.controller status --run RUN` read-only dashboard: phase/stage, update k, elapsed, per-deck
  status/quota/p/latest monitor vs MCTS, last train RSS and losses, log paths. Must work while running.
- `freeze` subcommand: copy the latest completed model to `out/frozen-models/<k>/` and write the current
  graduated list, for S3 (S3 itself is a separate later script; not in scope).

## Tests (CTest/pytest beside owners)
Seed determinism & independence across decks; quota allocation (min floor, budget sum, graduated floor); teacher
taper by per-deck age; replay window per deck; lifecycle transitions; train.py mixed batch + dup-fight error;
an end-to-end smoke with 2 decks, tiny budgets (sims 16, teacher sims 64, 2 fights/deck, 2 updates, wave_every 1,
1 epoch, CPU) using the frozen worker `runs/schema=combat_v4/date=2026-10-05/id=single-deck-fresh-final/out/frozen/pv_worker`
in `scratch/` — **use at most 2 workers for the smoke and only when no other pv_worker play is running**
(check `combat_loop.active_play_workers()`; S0 screen currently uses 10 — wait or ask me).

## Hard rules
≤10 gameplay workers total; never more than one gameplay stage at once. Don't modify frozen workers or
existing apps' default behaviour. No DuckDB fan-out; RAM 15 GB. Report back with: files changed, test results,
smoke output path, anything you were unsure about. Ask me rather than guessing on research semantics.
