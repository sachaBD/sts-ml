# Elite specialist: data and test plan

**Goal:** A value-net specialist that beats guided-rollout MCTS on Act 1 Ironclad A20 elite fights, with **20,000 simulations and 8 particles for both**. This is a fixed-fight experiment, not a full-act evaluation. **Both candidates fell short on validation.** The final-test fights have not been played.

## What is already available

- Baseline fights: `combat_v3/2026-09-26/act1-all-bosses-a20-scaled-search` (teacher, scaled budgets, one random move per fight).
- Extra fights: `combat_v3/2026-09-27/ab-selfplay-gen1-rest` (gen0 net replays of up to 330 source fights per non-boss encounter, 20k simulations, no random move; elite rows only here).
- Initial model: `value_net_v1/2026-09-27/ab-gen1` (gen0 trained on baseline buckets 6–9; gen1 fine-tuned on gen0 self-play from bucket 4). No training/selection from buckets 2, 3 or 5 for this checkpoint.
- [DATA_AUDIT.md](DATA_AUDIT.md): measured elite counts and a rough opening-deck diversity profile; [ROW_AUDIT.md](ROW_AUDIT.md): eligible label counts. Regenerate with `PYTHONPATH=python .venv/bin/python experiments/act-1-combat/2026-09-28-elite-specialist/audit.py` and `.../rows.py` respectively.
- Frozen fight-ID lists: `validation_fights.csv` (150 per elite) and `final_fights.csv` (500 per elite). Selected by SHA-256 of source episode ID within each encounter, **without inspecting outcomes**; the rest of the bucket-5 and confirm fights are not part of this planned evaluation.

## Partition rules (entire runs, never individual rows)

| Use | Baseline `run_seed % 10` | Additional replays | Reason |
|---|---|---|---|
| Train | 4, 6, 7, 8, 9 | gen0 self-play of bucket-4 sources | More diverse source fights; all underlying runs already eligible for initial model training |
| Validation / choose mixture and checkpoint | 5 | None | New run seeds and decks for this checkpoint; keep separate from training |
| Final comparison | 2, 3 | None during training or model selection | Previously reserved confirm runs, not used by ab-gen1 |
| Exclude | 0, 1 | None | Used in earlier model-development evaluations |

Only `category = 'elite'` and the three encounters Nob, Lagavulin and Sentries are in scope. Training has **7,536 baseline fights** (2,448 Nob / 2,517 Lagavulin / 2,571 Sentries), plus **990 gen0 replays** of 990 distinct bucket-4 source fights (330 per elite). There are **108,366 terminal-eligible decision rows** across those training fights; 146 baseline fights have no eligible row after their random move. The 1,547 baseline bucket-5 fights form the validation source pool; the 3,069 bucket-2/3 fights form the final source pool. Replays inherit the split of their **source** fight, not their newly assigned run seed. The audit verified all 990 replay source IDs are bucket-4 training fights and none are in held-out buckets. Prevent a frequently replayed deck from dominating training by weighting/sampling at the source-fight level. The audit's opening-card multiset is a proxy, **not** proof of distinct decks.

## Training attempts

- **v1 (`elite-terminal.toml`):** ab-gen1 initialization; train on 108,366 terminal-eligible decisions, excluding teacher rows at or before its random move. 6 epochs, best at 5. Equal total training weight per elite; fight weight grows as square root of its row count, and a replay shares its source fight's weight. Trainer-internal validation is from training seed buckets; bucket 5 remains external. **Validation failed:** [VALIDATION.md](VALIDATION.md) — net −1.28 HP-eq/fight (95% interval −2.50 to −0.09) vs MCTS, especially Sentries −4.21. This model is rejected, not run on final fights.
- **v2 (`elite-shift.toml`):** same source fights and initialization, but include teacher child search states as well as played decisions. The teacher's targets are shifted 50% towards the fight outcome after its random move; child states keep their teacher ordering. Gen0 replay decisions use the same target rule. Same encounter/source weighting. Best checkpoint epoch 1. **Validation fell short:** [VALIDATION-SHIFT.md](VALIDATION-SHIFT.md) — net −0.96 HP-eq/fight (95% interval −2.07 to +0.11), Sentries −2.63 (−4.85 to −0.44). Rejected; no final test.

No new replay data is assumed necessary at the outset; additional resamples can add outcome variability but **not deck diversity**. If deck coverage is thin, prefer new source fights over more copies of existing ones.

## Test protocol (to finalize before compute)

Replay held-out **complete starting fights** from the baseline, with the same source fights, combat start and search settings for the candidate and MCTS; no random moves. Use the frozen **500 per elite** from buckets 2–3 for the final comparison (1,500 paired fights); 150 per elite from bucket 5 can be used for candidate selection. Each source fight must be played once by each agent with the *same replay-start RNG/HP*, not independently resampled starts. Keep run seeds grouped when computing uncertainty; report per-encounter and combined paired HP-equivalent, win/death rates, and potions. An aggregate improvement with a material loss on any elite does **not** meet the goal. Validation fights cannot support the final success claim. Do not pick or drop final fights based on their outcome.

Validation replay uses `validation-mcts.toml`, `validation-net.toml`, and `validation-shift.toml`; the replay tool rebuilds the original fight from its run seed and prior actions, and checks the opening state. Each paired start matched on seed, encounter, HP and max HP in `compare.py`. All validation results are model-selection evidence only. Neither candidate merits the separately reserved 1,500-fight final test. **Next proposed direction:** new *diverse-source*, no-random-move elite games with 20k guided-rollout teacher search; current baseline has 5k teacher labels and a random move. Obtain approval for the compute before generating these fights.
