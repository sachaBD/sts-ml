# Phase 5: rollout-search expert iteration on fresh full combats

Run `runs/schema=combat_v4/date=2026-10-06/id=rollout-expert-iteration-v1/`: report `out/REPORT.md`, plus manifest, ledger,
frozen snapshots and logs/{smoke,main}.log. Code: `ei.py`. Executor single-fight-opus.

Two arms, both initialised from rescue update3 (same weights and AdamW moments):
- control collects with L0, strong with L0.5;
- each update both play the same 200 fresh full-start combats (600 seeds in total, excluded from all prior and final sets);
- 3 updates; training is the Stage2 train_fixed setup (1000 steps × 32 old + 32 new; old pool of 1432 fights = original
  1000 + Stage2 rescue 432); value target is the actual outcome; endpoint is update 3 only.

Endpoint wins on the 100 dev starts (2026-10-06 23:15; 1600/1600 games, 0 caps/errors, 21.9 min):

| model | L0.5 | L0 |
|---|---|---|
| strong | 83 | 71 |
| control | 80 | 69 |

References: MCTS 74; untrained rescue 76 at L0.5, 66 at L0.

Paired comparisons:
- strong-L0.5 vs control-L0.5: +3pp (5 vs 2 discordant, p=.45), not established.
- strong-L0.5 vs MCTS: +9pp (CI +1.1 to +16.9, 13 vs 4, p=.049).
- strong-L0.5 vs untrained rescue L0.5: +7pp (8 vs 1, p=.039).

The L0 cells changed little, so training did not amortise the search gain into the network.

Caveats: development starts reused across phases, several comparisons made, not confirmation. Fresh confirmation of
strong-L0.5 is pending Astra's decision.
