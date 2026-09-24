# Rundeck slime-v6: Slime Boss (A20 Ironclad), bootstrap → gen0 → DAgger gens

Status: **accepted plan, not started.** 2026-09-24 UTC. Results go in `REPORT.md`.

## Starting point (queried 2026-09-24)

- Bootstrap Slime fights: **458**, about 52k rows (act1-a20, -1, -2, -5, -6). The teacher won 71%, with the random move on.
- All `value_net_v1` runs have been deleted (slime-value-4 and -5). `dagger-gen1-slime-v5-001` can't be reused, so this is a fresh lineage.
- `act1-a20-3` is empty (overwritten).

## Terms

- **Teacher:** guided-rollout MCTS, 15k simulations. Fixed.
- **Model:** the value net (`value_net_v1`). The agent is the same MCTS with value-net leaves.
- **gen0:** trained on bootstrap rows.
- **gen k:** DAgger-corrected ("corrective fine-tune").

## Steps

1. **Bootstrap** `act1-a20-7`, `seeds = 1250`.
   - Target is about 800 Slime fights in total (see "Sizing").
2. **Compact** act1-a20-7.
3. **Train gen0** `slime-v6-gen0` on all bootstrap Slime rows.
   - 20% of fights are held out, about 160. These are the test fights for every generation.
4. **Evaluate gen0** with value_play, random move off, on the test fights.
   - `slime-v6-teacher-play` (`leaf = guided_rollout`): run once. This is the fixed baseline.
   - `slime-v6-gen0-play` (`leaf = value_net`).
   - compare_fights: gen0 vs teacher.
5. **DAgger round k** `slime-v6-dagger<k>`: the latest model plays **all eligible gen0 training fights** (about 640).
   - The teacher labels every position the model reaches.
6. **Train gen k** `slime-v6-gen<k>`.
   - Starts from **gen0's weights**.
   - Data: bootstrap plus dagger1..k.
   - gen1 only: train with `correction_weight` 0.25 and 0.5. The one that plays better sets it for later generations.
7. **Evaluate gen k.** value_play as in step 4, then compare against the teacher and against gen k-1.
8. **More generations:**
   - gen1: always.
   - gen2 and gen3: only if the previous generation beat the one before it (95% CI > 0) and is still below the teacher.
   - At most 3 generations.
9. **Afterwards (separate decision):** DAgger can at best match the teacher. Going beyond it means the value-net search generates its own training data, which needs code changes.

## Sizing

- **Bootstrap, about 800 fights.** Test fights are the constraint.
  - slime-value-5 vs teacher gave a 95% CI of about ±0.055 on 47 fights, so the per-fight SD of the paired difference is about 0.19.
  - Resolving 0.03 terminal value (about 2 HP) needs a CI of about ±0.03, so about 150 test fights.
  - 150 test fights / 0.2 ≈ 800 fights in total. We have 458, so we need about 350 more.
  - At 28% of seeds reaching Slime, that's about 1,250 seeds, roughly 18 min.
  - More bootstrap data for training isn't known to help: 55 → 290 training fights helped (v4 → v5), but beyond that is untested. DAgger supplies the targeted data.
- **DAgger, all eligible fights (about 640 per round).**
  - The test set doesn't limit it, so more is better. A bigger round makes a clearer test of whether DAgger helps.
  - Estimated time: round 1 about 51 min. Later rounds are the same size and may go over 1 h.
  - It gives about 18k correction rows per round, against about 72k bootstrap training rows.
  - Each round replays the same fights with a new learner, which reaches different positions, so no sampling is needed.
  - Agreed with the user. The trainer only accepts DAgger fights from gen0's training set, so new fights would need code changes. The cost is limited start variety (about 640 starts). The only fix for that is a bigger bootstrap before gen0.

## Hyperparameter changes (everything else unchanged)

| where | change | baseline it differs from |
|---|---|---|
| gen0 | epochs 7 → 8 | slime-value-5 (lr 1e-3, batch 128, width 64, blend 0.5) |
| gen k | starts from gen0's weights, not gen k-1 | `apps/value_train/slime-dagger-gen1.toml` (lr 1e-4, 7 epochs) |
| gen1 | `correction_weight` tried at 0.25 and 0.5 | same file (0.5) |
| DAgger | count: all eligible fights (100 before) | `apps/dagger/slime.toml` |
| eval | `random_move = false` | value_play default |

## Code change (impl-24)

- **Problem:** `value_play` and `dagger` reject a value net whose inputs include a value_net_v1 run or DAgger runs. Every gen ≥ 1 model has those, so neither app can evaluate or collect with it.
- **Fix:** a shared `bootstrap_inputs(run_id)` returns only the plain bootstrap combat_v3 inputs, and both apps use it. There is one path and no fallback.
- **Needed from step 5, round 2 onwards** (the gen1 evaluation in step 7 is the first use).
- **Done:** commit 114a6ac. Checked: new tests pass (`PYTHONPATH=python:tests .venv/bin/python -m unittest test_dagger test_value_play test_value_training`). The one error, `test_split_is_deterministic_and_disjoint`, is a stale test: its fixture rows have no `run_seed`, and the split groups by it. The code is fine; it doesn't affect this plan.

## Guardrails

- One job with 10 workers at a time. Single-threaded trainings can run in parallel.
- Configs go in `rundecks/slime-v6/configs/`. Write each one after its inputs exist, because run ids contain the UTC date.
- Never train on `*-play` runs. The test fights are never passed to DAgger (the app enforces this).
