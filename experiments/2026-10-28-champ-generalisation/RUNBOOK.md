# Thousand-deck generalisation — night of 2026-10-28

## Goal
Train one A20 Champ agent on ~1,000 distinct human-derived deck compositions;
measure transfer to separate unseen compositions, not merely new seeds of training decks.
Keep the 64/2/2 architecture; warm from fifty-deck checkpoint40. All50ancestor decks
are training-exposed and must be excluded from development/final test.

## Planned recipe
- ~1,000training decks,5games/deck/update (~5,000total), not100games/deck.
- Ten-update disk replay (~50,000games); all retained row addresses eligible.
- Deck-balanced effective batch2048;500optimizer steps/update.
- Length-bucketed attention-bounded microbatches; one weighted gradient/optimizer
  update per effective batch. No state truncation or dropping slow games.
- 20%recordedwall-time validation. Separate legacy50, additional training-monitor
  decks, and unseen development families; final families never enter the live loop.
- Primary generalisation comparisons use family-clustered95% bounds versus the
  starting checkpoint. Historical MCTS20k/PUCT are labelled by policy and ruleset.

## Safety
No current training job is live: previous run failed20:17UTC;checkpoint40safe.
Bounded automatic recovery is for recognised CUDA failures only: fresh process,
latest committed checkpoint/optimizer/RNG, interrupted stage archived, at most2
retries. STOP wins over recovery. Unknown/deterministic/repeated failures escalate
rather than loop. Never count infrastructure failure as a game loss.
Maintain at least10GiB disk reserve. No simulator correctness reversions.
Do not modify or commit another agent's web.py/web.html edits.

## Audit
`inventory.py` scans929compactcombat_v4fight tables, not action/search traces.
Outputs:`scratch/champ-generalisation-audit/{summary,families,policies,agents}.json`.
Historical results are heterogeneous: keep frozen policy/run identities separate.
1,018staticallycompatible existingpoolfamilies; native qualification still required.
535compatible human-source compositions outside that pool and current50lineage.
The JSON-only legacy outputs have not yet been exhaustively scanned; do not claim
that parquet coverage is every historical result.

## Execution status
Preparation only. No thousand-deck training job launched yet.
Exact frozen split, manifests, qualification results and recovery/launch commands
will be recorded here before launch.
