# Human imitation (the competitor's picker)

**Status: reference only.** `../sts_ml` is a competing, currently stronger agent. It is not a component of ours.

Source: `../sts_ml/mined-report.md`, not independently verified against its code.

## What it does

- **Hybrid agent:** a human-choice network scores supported strategic screens (card rewards, boss relics, single-card upgrades and removals). A second network adds residual logit corrections. Map routing is the native simulator expert plus hand-coded Heart-key rules. Combat is native search.
- **Architecture:**
  - Card/relic embeddings (identity, upgrades, counts) feed a 1-layer, 4-head Transformer over the inventory, which is pooled and combined with screen and numeric features.
  - A **2-layer, 192-wide GRU** carries memory of previous supported strategic choices.
  - Each candidate cross-attends to the inventory. A screen-specific head scores candidate + GRU context + attention output. Skip is a real option on card rewards.
- **Observations:** deck/relics, candidates, screen, act/floor, boss. **HP, gold, map and counters are masked**, because the human logs lacked them. The GRU likely partly stands in for this missing state and for archetype commitment.
- **Training:** teacher-forced cross-entropy on ~655k decisions from ~24k human Heart-win runs. The checkpoint was selected by held-out likelihood. This is imitation, not RL; the value heads are unused.
- **Result:** the whole agent won 13/1,000 A20 Heart simulator runs (~1.3%). The picker's contribution is not isolated.

## Takeaways for us

- **Its weakness is the target, not the structure.** It learns "what a human winner picked", not win probability for its own agent, and it cannot see HP/gold.
- **Ideas worth reusing in our own picker network (CARDS step 5):**
  - Candidate-vs-deck attention scoring, with skip as an action.
  - A shared scorer over variable-length candidate lists, rather than fixed output slots.
- **The data is not available locally:** only weights are present. We would not use its weights in our agent anyway.
