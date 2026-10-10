# Final confirmation: rollout-search expert-iteration candidate vs MCTS 20k

Run `runs/schema=combat_v4/date=2026-10-06/id=demon-form-rollout-confirmation-v1/` (out/REPORT.md, manifest.json, fights.json,
ledger.jsonl, frozen/; logs/main.log, audit-text-hits.txt, audit-parquet.json). Code: `phase6/confirm.py`, `phase6/audit_parquet.py`.

**Candidate:** Phase5 strong/update3, frozen pv_worker 58c5c4ab `play MODEL 2000 --rollout-mix 0.5`, greedy.
- model.onnx 49eeb12c…; .data b5c08a33…; model.pt a93944e1…

**Reference:** the same worker running `teacher 20000`, played fresh.

**Seeds:** the ORIGINAL reserved 600 (`single-deck-demon-form-v1/out/final-reserved.parquet`, demon-form-fresh-v1:final:0–599;
Demon Form deck, 34/52 HP, no potions). These seeds are now CONSUMED.
- Text audit: grep over every repo text artifact for the prefix and all 600 seeds: 0 hits.
- Parquet audit: 1051 parquet files modified after the reservation, checking fight_id/seed/start.seed: 0 hits.
- Not scanned: parquet files older than the 09:08 reservation (the namespace did not exist before then).
- Disjoint from the overnight starts (12,804 seeds, 0 overlap).
- Deviation: Astra's later message, choosing a NEW 600 namespace and keeping the reserved seeds untouched, crossed with
  the launch. The run had already started on the audited reserved seeds.

**Result (2026-10-06 22:42–22:57 UTC; 1200/1200 games complete, 0 caps/errors):**
- Candidate 485/600 vs MCTS 441/600.
- Paired gap +7.33pp (95% CI +4.62 to +10.04); 58 candidate-only vs 14 MCTS-only wins; exact McNemar p=1.6e-7.
- Preregistered success (lower bound > 0): **MET**.
- Six 100-seed batches, descriptive only: gaps +9, +5, +6, +5, +12, +7pp, all positive.
- Cost: candidate 7.41 s/game (median 7.03) vs MCTS 7.81 (median 7.39) on 10 concurrent workers; load variability not quantified.

**Caveats:**
- A single training RNG and a single fixed loadout (Demon Form 34/52 vs A20 Champ).
- The result is conditional on this deck; it is not proof of a general recipe.
