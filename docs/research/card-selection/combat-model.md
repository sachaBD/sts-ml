# Combat outcome model: first implementation brief

## Contract

Predict the distribution of combat outcomes from **pre-combat public state plus encounter identity**, marginalising over combat RNG and search randomness, for an identified combat policy/budget.

First deliverable: a batchable Python model useful for the gauntlet, not a complete macro transition simulator. Start with survival probability and ending-HP distribution conditional on survival. Potion count can be an auxiliary target if cheap; count alone is not sufficient to continue a run faithfully. Do not silently call this a complete transition model.

Define the endpoint explicitly: recommended eventual contract is after combat resolution/end-of-combat effects, before rewards. Establish where existing `final_hp` is measured, particularly Burning Blood, before interpreting it as macro HP. Do not double-count healing. Max-HP changes can make ending HP exceed starting max HP.

## Evidence inspected

- `agents/combat/value/README.md`, `agents/combat/value/deep_sets_v3.py`: typed card/monster/potion/relic tokens, summed pools, residual head. Existing heads estimate win probability and conditional mean HP/potions, combined using a combat score. These are useful building blocks, not the right unchanged interface for a pre-combat model.
- `environments/combat/encoding_v4.cpp`: potion identities/effects, combat relic identities/counters, player and monster statuses. Encoding is computed from BattleContext; some features already reflect battle initialisation.
- `environments/combat/schema.py`, `runs/README.md`: trajectory/child rows, repeated fight outcomes, source lineage, starting HP/max HP, optional v4 fields. Terminal potion **count**, not terminal inventory; no explicit complete post-combat persistent state.
- Actual parquet spot check: one compact file in `combat_v3/2026-09-26/act1-all-bosses-a20-scaled-search` has no v4 relic fields. Do not assume schema documentation means old data contains them. No full inventory performed yet.
- The all-bosses source run's manifest has interrupted/failed status; this alone does not establish whether completed fight records are unusable.
- `apps/gauntlet/gauntlet.py`, `worker.cpp`, `act1.toml`: already evaluates offered cards and skip with shared future-offer/fight seeds, SimpleAgent future picks, elites and known boss. It holds HP/relics/potions at the reward state and uses a weighted HP-loss/death-penalty score. This is a capability benchmark, not a faithful future run.

## Input recommendation

Use the permanent deck multiset (card identity, upgrades and permanent mutable properties), current/max HP, relic identities and relevant counters, potion identities/capacity, encounter identity, and any additional pre-combat variables actually affecting the fight. Fixed A20/Ironclad/Act1 need not be learned categorical inputs initially.

Start with a small Deep Sets encoder for deck/relic/potion tokens plus an encounter embedding and MLP head. Reuse encoding conventions where valid; create a new topology kind rather than changing a frozen existing one. No attention requirement, C++ export, large architecture sweep or pretrained-weight requirement for iteration one.

Do NOT feed opening hand zones, realised enemy HP/intent, hidden RNG, search values, outcome-derived features, or run seed into a pre-combat predictor. They are unavailable at card selection time. Flattening first-decision card zones is not automatically sufficient: initialisation may add cards or alter upgrades/costs and relic counters. Prefer reconstructing the pre-combat GameContext using existing replay machinery; implementation should first establish cost and fidelity. An explicitly restricted first-decision predictor is a different target and must not masquerade as the macro interface.

## Training data

- One labelled example per fight execution, identified by source run plus episode (episode IDs can repeat across replay runs). Source run seed groups all descendants for splitting.
- Use realised outcomes, not root values or blended search targets. Never attach played outcomes to counterfactual child rows.
- Exclude oracle data. Identify random-action exploration fights and learner-played DAgger data; these do not describe a pure frozen teacher policy.
- Identify actual policy/checkpoint/search budgets from manifests/config provenance. Start with a coherent useful subset; do not require all data to share a single budget if an explicit encounter-dependent policy is intended. Do not silently pool incompatible policies.
- Audit missing relic/potion/deck fields before agents.combat.value. Replay extraction may be necessary; do not infer absent identities from outcome correlations.

Suggested first output parameterisation: win logit plus conditional HP histogram with documented bins/support and overflow handling. Train BCE plus conditional categorical NLL. A simpler mean-HP head is a useful diagnostic baseline, but not a stochastic transition model. Choose the simplest supported output with the implementer; the distribution family is not settled.

## Deliberately deferred

Full joint HP/potion/permanent-effect transitions; macro planning; adaptive simulation allocation; model ensembles; full-run confirmation; broad counterfactual data generation. Document omissions that matter for Feed, potion generation, permanent card growth and relic counters rather than fabricating updates.

## First implemented cohort / endpoint (implementation-session report)

Extraction artifact: `combat_transition_v1/2026-09-29/act1-eval-mcts-a20`, from the corresponding 1,000-seed combat evaluation. Reported 6,888 fights: 6,217 replay successfully, 180 diverge, 491 follow a divergence and are not replayed. These counts are implementation-reported, not independently audited here. Similar marginal win rates do not exclude card/relic-dependent replay selection bias.

Policy `mcts_gr_catbudget_v1`: guided-rollout MCTS, 8 particles, budgets easy 500 / hard 2,000 / elite and event 5,000 / boss 15,000; no oracle or random moves. Initial split: source-seed buckets 4–9 train (3,757 replayed fights), 0–1 development (1,238), 2–3 excluded. This is a bounded initial cohort, not a claim that all available data has been used.

Extractor reports `battle_final_hp` precedes `exitBattle`; post-state HP includes end-of-combat effects, including Burning Blood. First model may target **battle_final_hp** for compatibility with the current gauntlet. This is explicitly a battle-outcome predictor; macro continuation requires the post-exit transition. Both endpoints and full persistent state are retained. Do not assume all post-state effects reduce to adding six HP.

Recommended initial conditional HP distribution: coarse fixed bins with explicit overflow rather than many sparsely supervised one-HP classes. Keep this a single small experiment, not an architecture sweep. Report tail handling and observed overflow. Survival supervision is scarce outside bosses/elites in this cohort.
