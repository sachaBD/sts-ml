# Outcome UI notes

- Built in `gui/` (user's request, instead of `apps/outcome_ui/`). See `gui/README.md`.
- Verify (`gui/verify.py`): 200 natural rows from `...-s0/out/augmented-predictions.jsonl`, matched to natural rows by
  (run_seed, encounter, start_hp, final_hp), max |Δ p_win| = 1.2e-7, max |Δ mean_hp_if_win| = 1.5e-5 (float32 noise
  from different batch padding). Header id tables match the names in all 4,995 natural rows.
- Legacy comparison checkpoint not wired in (optional in the brief).
- Added: "Real vs predicted" section (held-out natural fights, run_seed % 10 in {0,1,8}: per encounter/group + reliability
  bins), the real outcome shown for an unedited preset, and a "Starter deck" preset (A20, 68/75 HP, 2 potion slots).
  Presets now come only from held-out natural fights.
