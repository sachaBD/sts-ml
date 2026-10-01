# Brief: small win-rate UI for the pre-combat outcome model

## Goal
A local UI where the user picks a deck (cards and upgrades), relics, potions, HP/max HP and an encounter, then sees:
- P(win), with an uncertainty band;
- the distribution of final HP if the fight is won;
- optionally, the predicted change from adding one card (take vs skip).

Ironclad, A20, Act 1 only. The UI is local and single-user. Keep it small: a single Python file is fine (Gradio or
Streamlit, or FastAPI with one HTML page).

## Model (read-only, never retrain)
- **Checkpoints:** `runs/schema=combat_outcome_v1/date=2026-09-30/id=tpair1-co2-w32-h64-l1-d30-lr.001-s{0,1,2}/out/augmented.pt`.
  These are 3 training seeds of the recommended model; see `experiments/topology-sweep/REPORT.md`.
  - Use all 3 as a small ensemble. The mean is the estimate and the spread across seeds is a rough model-uncertainty band.
    Label it as rough: it is 3 seeds and is not calibrated.
  - Optional comparison: legacy `id=card-outcomes-s3c-model-supp*/out/augmented.pt`.
- **Loading:** each checkpoint is a dict with `kind`, `args`, `state_dict`, `encounters` (name→id map) and `centers` (HP bin
  centers). Build the network with `KINDS[kind](**args)` from `apps/combat_transition/train_marginals.py`, which imports the
  kinds from `python/sts_combat_rl/topology/`. See `apps/combat_transition/eval_marginals.py` for a complete load-and-score
  example.
- **Encoding:** use `apps/combat_transition/train.batch(rows, encounters, device)`. Each row is `{"encounter": name,
  "pre": {...}}`, where `pre` holds `deck`, `relics`, `potions`, `hp`, `max_hp` and `potion_capacity`. Copy the exact
  element formats (card id, upgrades, misc; relic id and data) from a real row. `train_marginals.dataset()` or the
  natural/synthetic parquet runs listed in `experiments/card-outcomes/data/through-s3c.txt` provide real rows. Card, relic and
  encounter names/ids come from the same encoders the generator uses; find the id↔name tables in the repo rather than
  hard-coding them.
- **Outputs:** the forward pass returns `(win_logit, hp_logits)`.
  - `P(win) = sigmoid(win_logit)`.
  - `P(HP bin | win) = softmax(hp_logits)`, with 21 bins: 1–5, 6–10, …, 96–100, and >100.
  - Expected score (HP if won, else 0) = `P(win) · Σ softmax·centers`.

## Must
- Start by pre-filling from a real natural pre-state, e.g. the deck before a boss fight, so the user can edit from a valid
  state.
- Only offer encounters in the checkpoint's `encounters` map, grouped as easy / hard / elite / boss.
- Take-vs-skip view: the base deck, plus one row per candidate card showing the Δ in P(win) and in expected HP.
- Show a caveat line in the UI:
  - this is a development model;
  - elite/boss card effects explain only about 20–35% of the real effect variance;
  - a boss P(win) near 0.5 is where the model is least reliable.
- Do not modify training code, run dirs or `topology/`. Put new code in `apps/outcome_ui/` and notes in `experiments/outcome-ui/`.

## Verify
- Reproduce `p_win` for a few rows of `…/tpair1-co2-w32-h64-l1-d30-lr.001-s0/out/augmented-predictions.jsonl` through the
  UI's code path, to within about 1e-6.
