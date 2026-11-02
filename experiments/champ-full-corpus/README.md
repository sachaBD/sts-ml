# Champ full corpus

## Objective

Train an autonomous Ironclad A20 Champ agent from a cold start on the full usable Mega Crit corpus, including
human wins and losses. Improve performance on unseen deck compositions without losing performance on the legacy
50 decks. One training run: evaluation uncertainty does not measure variability across training runs.

## Agreed allocation — 2026-11-02

**Targets, not yet frozen assignments. No training launched.**

| Partition | Decks | Approx. share | Fixed fights/deck | Evaluation fights |
|---|---:|---:|---:|---:|
| Training | ~7,942 | 80% | — (sample during training) | — |
| Legacy validation | 50 | 0.5% | 10 | 500 |
| New validation | 950 | 9.5% | 2 | 1,900 |
| Final test | 1,000 | 10% | 10 | 10,000 |
| Total | ~9,942 | 100% | | 2,400 validation |

- Validation contains **1,000 decks total**, including all 50 legacy decks.
- Preserve legacy loadouts, HP and historical seeds; select ten existing starts per deck deterministically.
  Legacy reference: `runs/benchmark/champ-50-inputs/frozen/`. Verify all 50 before freezing.
- New validation and final starts use human HP, fresh fixed seeds and sampled potions. Freeze the realized starts
  once, not a fresh draw at each evaluation.
- Assign new validation, final and training by **upgrade-agnostic card composition**, preserving card multiplicity.
  A composition must never cross partitions. Exclude all legacy compositions from new validation and final.
  Training must also exclude legacy compositions: retention decks are validation-only in this cold-start run.
- Counts may change slightly to keep composition groups intact. The table assumes legacy decks can be matched to
  the candidate pool; only the original 25 have been checked so far. Record actual counts and denominators when frozen.
- No selection of new decks by human or agent win rate. Human outcomes are provenance, not selection criteria.

## Evaluation and checkpoint selection

Report legacy retention and new-deck generalisation separately. **Primary metric: macro win rate over the 950 new
validation decks**, averaging each deck's fixed-fight win rate. Do not mix it with the legacy score for checkpoint
selection. Two fights per new deck are deliberately sparse: individual-deck estimates are noisy, breadth is the goal.

Report paired changes on the same starts and uncertainty clustered by composition. Intervals are conditional on
this training run and are not adjusted for repeatedly inspecting validation.

Choose the checkpoint with the best committed primary validation score; ties choose the earlier checkpoint.
Freeze its identity before evaluating final. Final starts must never enter replay, routine evaluation or further
checkpoint selection. Compare the chosen checkpoint and cold-start checkpoint on the same reserved final fights.

## Candidate dataset — unsplit

`runs/benchmark/megacrit-ironclad-a20-candidates-20261102/decks.jsonl`

- 14,169 unique human runs reached Champ across four A20 Ironclad pulls.
- Conversion retained 10,368; 3,797 failed reconstruction rules and four had invalid HP.
- 9,942 candidates passed native starting-state import/encoding after 426 Dome/Shard exclusions.
- Native check validates starting-state capabilities, not all effects reachable during play.
- Converter approximations remain: unknown relic counters and bottled assignments; unknown card misc values
  (353 converted decks); multi-upgrade Searing Blow represented as boolean upgraded (314 converted decks).
- Source and detailed counts/hashes:
  `runs/schema=pv_starts_v1/date=2026-11-02/id=ironclad-a20-all-20261102-v2/out/dataset-audit.json`.
- Filtering script: `scratch/megacrit-preparation/filter_starts.py`; its output directory contains `summary.json`
  with source/binary hashes and `rejected.jsonl` with reasons. Preserve this script with the frozen dataset.
- Earlier converter-generated 96/2/2 partitions are **superseded**. Build the agreed experiment from the unsplit pool,
  not those `validation.jsonl` / `final.jsonl` files.

## Next gate

Audit all 50 legacy matches and reconstruction limitations; prepare and freeze composition-disjoint partitions,
input hashes and selection provenance. Then record the cold-start config, worker revision and evaluation schedule.
No checkpoint warm start. Training HP jitter and potions are intended; exact training settings and launch budget
remain to be frozen. Do not launch training as part of dataset preparation.
