# Recency continuation — completed result

Run: `runs/schema=combat_v4/date=2026-10-07/id=expert-champ-recency-v1/`.
Completed 2026-10-07 21:01 BST; about 57.5 minutes elapsed from launch, ten gameplay workers. Raw ledger independently recounted: 3,402/3,402 completed results, no caps/errors.

## Independent final

Selected update4 versus frozen Phase5 strong/update3 incumbent on 600 fresh paired starts:

- Candidate 482/600 (80.33%), incumbent 477/600 (79.50%).
- Gap +0.83 percentage points; approximate paired 95% interval -1.10 to +2.77 points.
- Candidate-only wins 20; incumbent-only wins 15; same outcome 565/600.
- Exact McNemar p=.500. **Improvement not confirmed.** Keep incumbent as the best-supported benchmark; retain all candidate artifacts.

Update4 was selected from five monitoring checks: 161,159,161,167,161 wins /200 versus incumbent158. Its selected +4.5pp monitor estimate shrank to +0.83pp on final; consistent with selection optimism. Monitor results are not independent confirmations. No uniform-replay training control, so this is not a causal test of recency weighting.

This continuation (1,000 fresh fights, 5,000 optimizer steps) did not establish a material strength gain. It does not prove a hard performance ceiling or that more/different training cannot help. One deck and one training RNG.

## Compute lesson and standing session preference

User requests **at least 3:1 training-to-evaluation compute in future runs**. Do not repeat evaluation after every 200-fight training batch. Budget by measured cost, not only game counts; account explicitly for incumbent references and independent final confirmation as well as intermediate monitoring. If a small exploratory run cannot afford confirmation within that allocation, agree an explicit exception or defer confirmation rather than silently violating the ratio.

Actual pilot gameplay: 1,000 collection, 1,200 monitor, 1,200 final, 2 smoke. Summed subprocess elapsed worker-seconds: collection9,472.5; monitor11,720.0; final11,050.2; smoke11.2. These are not CPU-time measurements; system-load uncertainty unquantified. About29% of measured gameplay time went to collection, excluding neural optimization/export. This was over-evaluated for an exploratory training run.

## Recommendation

Do not spend another large confirmation budget proving whether this sub-percentage-point estimate is real. Keep recency as an unproven practical replay choice, not an established strength improvement. Next work should prioritize improving the policy-improvement step or testing a cheap bounded mechanism, with sparse monitoring and the explicit >=3:1 compute budget.
