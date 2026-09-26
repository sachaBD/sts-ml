# act-1 gen0 report: one general value net vs the MCTS teacher

**Verdict: the current topology (Deep Sets, width 64) fits act-1 teacher play.** One net trained on every act-1
fight falls short of the teacher on every encounter, but never by more than the slime specialist's known apprentice
gap. On Slime Boss it plays exactly as well as the specialist. No encounter is an outlier, so no topology, depth,
conditioning or per-fight head changes are needed at this stage. The next step is the self-play loop that took slime
from −3 to +4 HP-eq against the teacher.

Setup, split and units: `README.md`. HP-eq = Δ terminal value × (55 + max HP); 1 HP-eq is 1 HP, a potion is
4 HP-eq, and a death costs all remaining HP + 35.

## Headline

| | result |
|---|---|
| All 4,810 dev fights | **−1.2 HP-eq per fight** (95% CI −1.5, −0.8). Wins 93.0% → 91.6% |
| Per act-1 run (fights weighted by how often a run meets them) | **−6.5 HP-eq per run** (−8.3, −4.7). About 2.4 of this is the boss and 1.3 is the elites |
| Slime Boss: general net vs slime specialist, same 999 fights | **−0.1 HP-eq** (−1.6, +1.4). Wins 698 vs 698 (82 / 82 swaps) |
| Reference: slime specialist vs teacher | −2.8 HP-eq (−4.3, −1.3). Every encounter below is at or inside this |

## Per encounter (net − teacher, paired on the same dev fights)

| encounter | fights | Δ HP-eq per fight (95% CI) | win % teacher → net | swaps net / teacher (McNemar p) | HP lost when both won: teacher → net | val R² |
|---|---:|---|---|---|---|---:|
| boss/slime_boss | 999 | −2.9 (−4.4, −1.5) | 74.7 → 69.9 | 55 / 103 (<0.001) | 19.6 → 19.7 | 0.84 |
| elite/gremlin_nob | 250 | +1.0 (+0.0, +2.0) | 94.8 → 96.4 | 5 / 1 (0.22) | 21.9 → 21.4 | 0.82 |
| elite/lagavulin | 250 | −2.2 (−3.6, −0.8) | 92.4 → 88.4 | 1 / 11 (0.006) | 26.4 → 26.8 | 0.84 |
| elite/three_sentries | 250 | −1.9 (−3.7, −0.1) | 90.4 → 87.2 | 6 / 14 (0.12) | 27.3 → 27.7 | 0.84 |
| easy (4 encounters) | 250 each | −0.2 to −0.8 | 100 → 99.6–100 | ≤1 loss | within 0.7 HP | 0.85–0.95 |
| hard (10 encounters) | 124–247 | −0.5 to −1.3 | 97.6–100 | ≤2 losses | within 0.5 HP | 0.87–0.95 |
| events (4, pooled, descriptive) | 178 | −0.1 to +3.1, all CIs cross 0 | — | — | — | 0.81–0.90 |

Full table: `gen0-vs-teacher.md` / `.json`. Slime vs specialist: `gen0-vs-slime-specialist.json`.

What the table shows:
- **The gap is mostly in winning, not in HP.** On fights both players win, HP lost is within about 0.5 HP everywhere.
  The HP-eq gap on the boss, lagavulin and sentries comes from extra losses, the same pattern as the slime specialist.
- **Hallway fights cost about 0.5–1 HP per fight.** That is statistically clear but small.
- **Lagavulin is the weakest elite** (11 losses vs 1). It is still inside the slime reference gap.
- 22 intervals are shown, so about one could exclude the truth by chance. The gremlin_nob +1.0 is borderline and
  should not be read as a real gain.

## Training (value_net_v1/2026-09-26/act1-gen0)

- 1.82M rows (30,206 train fights, natural encounter mix), 10% of train runs held out for validation.
- 12 epochs, AdamW, lr 1e-3 with cosine decay to 0, batch 128, width 64, blend 0.5 labels. The checkpoint kept is
  the best on validation (epoch 11). Training took 45 min on 1 CPU thread.
- Validation MSE by epoch: 0.00471 → 0.00464 → 0.00457 → 0.00456 → 0.00450 → 0.00448 → 0.00459 → 0.00449 → 0.00449 →
  0.00446 → **0.00445** → 0.00446. Train MSE was 0.0027, so the gap is small and there was no real overfitting. Most
  of the gain came in epoch 1; after that the curve is flat, so more epochs or a wider net would add little on this
  data.
- Slime Boss validation MSE was 0.0073, the same as the specialist's best (0.0072). Sharing the net cost slime nothing.
- Per-encounter R² (1 − MSE / variance) was 0.81–0.95. Elites and the boss are lowest; see `training_history.json`
  in the run.

## Runs and compute

| run | what | time (10 workers) |
|---|---|---|
| `value_net_v1/2026-09-26/act1-gen0` | general net | 45 min, 1 thread |
| `combat_v3/2026-09-26/act1-teacher-dev` | teacher (guided rollout, 20k sims, 8 particles, no random move), 3,811 non-boss dev fights | 35 min |
| `combat_v3/2026-09-25/slime-v8-rollout-teacher-dev` | teacher on the 999 slime dev fights (reused, not re-run) | — |
| `combat_v3/2026-09-26/act1-gen0-dev` | net (value leaves, same budget), all 4,810 dev fights | 76 min |

The net search is slower than the teacher: 9.4 s per fight overall, against 5.5 s for the teacher on non-boss fights.
The confirm set is untouched.

## Code changes (uncommitted)

- `python/sts_combat_rl/training/train_value.py` and `apps/value_train/train.py` gained two new options. Both
  default off, so existing configs behave exactly as before:
  - `lr_schedule = "cosine"`: per-step decay to 0.
  - `keep = "best"`: keep the best-validation checkpoint.
- Every run now logs validation MSE per encounter each epoch, and writes the per-epoch history to
  `out/training_history.json` and the checkpoint json.
- In this directory: `split.py`, `analyze.py`, `configs/`.

## Next step

Self-play with terminal-value fine-tuning (expert iteration). The plan is in `README.md`, under "Next: gen1".
