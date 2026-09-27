# elite-v3: train and test the v3 net (full state + policy head) on Act 1 A20 elites

**Owner away ~8 h (from ~22:30 UTC 2026-09-27). Executor: orchestrator-27. Author: nn-consultant.**
Everything here is approved to run. Follow the stages in order and apply the gates exactly; log every launch,
finish and decision in `LOG.md`. When in doubt, stop the deck and write down why, rather than improvising.

## Goal

Beat guided-rollout MCTS (20k simulations, 8 particles) on a mix of the three Act 1 elites (Gremlin Nob,
Lagavulin, Three Sentries), paired on identical fight starts. Previous attempts (value-only nets `deep_sets_v2`)
lost: −1.28 and −0.96 HP-eq/fight on the 450 validation fights, Sentries worst
(`experiments/act-1-combat/2026-09-28-elite-specialist/`).

What is new here:
- **deep_sets_v3** sees what v2 could not: monster statuses (Sentry artifact, Nob enrage, Lagavulin asleep /
  metallicize), the monster's previous move, player statuses from potions, held potions and relics. Its value is
  the terminal score formula over win / HP / potions-kept heads.
- **Policy head + policy-prior search** (`leaf = "policy_net"`): the net's move priors at every search node (PUCT),
  its value at the leaves (sts_lightspeed policy-prior mode).
- **Current simulator** (Smoke Bomb disabled, Blood Potion fixed, no potion discards). The old baselines ran on the
  old simulator, so the MCTS baseline is re-run here on the same fights.

Primary metric: paired HP-eq/fight vs MCTS (`compare.py`; HP-eq = Δ terminal value × (55 + max HP), one potion =
4), overall and per elite, with 95% run-seed cluster-bootstrap intervals. Also win/death counts and potions kept.

## Ground rules

- Run from the repo root: `cd /home/sborowsk/project/sts_combat_rl`. The apps build `build/main` themselves.
- **One CPU-heavy app at a time**, except where a stage says to overlap (GPU training + one CPU app).
- Templates with `{{...}}` must be rendered first; the command prints the rendered path to launch:
  `.venv/bin/python rundecks/elite-v3/render.py configs/NAME.toml [--var best=... --var c_puct=...]`
  (it resolves `{{run:value_net_v1/ID}}` to the finished run and refuses unfinished runs).
- Compare runs with
  `PYTHONPATH=python .venv/bin/python rundecks/elite-v3/compare.py --baseline ID --candidate ID ... --output rundecks/elite-v3/results/NAME.md --json rundecks/elite-v3/results/NAME.json`
  (ids are the `[run] id` values; `mkdir -p rundecks/elite-v3/results` once).
- Never reuse a run id. If a run fails and must be repeated, append `-r2` to its id in the config and log it.
- Replays of stored fights may diverge from the stored fight on the new simulator; configs set
  `skip_diverged = true` so such fights are skipped and listed in the run's `summary.json`
  (`skipped_diverged`). Both arms of a comparison skip the same fights, so pairing holds. Log the count.
- Clock: `T0` = the time Stage 0 starts. The time boxes below are the plan; the gates use elapsed time since T0.

## Stage 0: preflight (≈15 min)

1. Record `git -C . rev-parse HEAD`, `git -C ../sts_lightspeed rev-parse HEAD`, `nvidia-smi`, `free -g` in LOG.md.
2. `make build && ctest --test-dir build/main --output-on-failure` and
   `PYTHONPATH=python:. .venv/bin/python -m unittest discover -s tests`. All must pass (including
   `tests/test_deep_sets_v3.py`: v3 encoder/loader/model parity and a policy_net search smoke test).
   **Gate:** any failure: stop, log, notify nn-consultant.
3. Smoke: `./apps/fight_resample/run.sh rundecks/elite-v3/configs/s0-smoke-resample.toml --scratch`
   (6 fights, 2k simulations, a minute or two), then
   `PYTHONPATH=python .venv/bin/python rundecks/elite-v3/check_data.py scratch/schema=combat_v3/date=*/id=elite-v3-smoke/out --min-fights 1`.
   **Gate:** GATE PASSED. (If all 6 replays diverged, log it and continue; Stage 1's gate decides.)

## Stage 1: teacher data (≈95 min, the longest stage)

`./apps/fight_resample/run.sh rundecks/elite-v3/configs/s1-teacher.toml`

5,100 training fights (1,700 per elite, hash-selected from training buckets 4,6,7,8,9), one resample each
(new fight RNG, HP ~ N(stored, 10), random potions), played by the MCTS teacher at 20k/8, no random move. Expect
≈1.1 s/fight wall on 11 workers (previous 20k elite runs: 450 fights in 9 min on 10 workers).

Then: `PYTHONPATH=python .venv/bin/python rundecks/elite-v3/check_data.py runs/schema=combat_v3/date=*/id=elite-v3-teacher/out --min-fights 3500 --max-skipped 0.35`
Save its output in LOG.md (potion / relic coverage matters for the report).
**Gate G1:** GATE FAILED: stop the deck, notify nn-consultant, write what happened in REPORT.md.
If it is still running at T0 + 2h15, do not kill it; note it. Stages 4/5 will then be dropped by their gates.

## Stage 2: train (GPU, ≈20–40 min) with the MCTS validation baseline in parallel (CPU)

Start these three together once G1 passes. The two trainings share the GPU, so run them one after the other.
The MCTS baseline can run alongside them.

- `./apps/value_train/run.sh rundecks/elite-v3/configs/s2-t1.toml`, then `./apps/value_train/run.sh rundecks/elite-v3/configs/s2-t2.toml`
  (t1: policy loss weight 0.05; t2: 0.25. Otherwise identical.)
- In parallel: `./apps/value_play/run.sh rundecks/elite-v3/configs/s3-v-mcts.toml` (450 validation fights, 10 workers, ≈10 min).

If t1 dies of memory (15 GB RAM): run `s2-t1-decisions.toml` (id `elite-v3-t1d`, decision rows only) instead,
skip t2, and use `elite-v3-t1d` wherever this plan says `elite-v3-t1`.
Log each training's best epoch and its final `v3:` metrics line (policy_validation_ce / top1, aux_keep MSE) and `validation_mse`.

## Stage 3: validation play (450 bucket-5 fights, 11 workers, ≈10–15 min each)

Run one at a time:
1. `s3-v-t1-value.toml`: t1 as a value-only leaf. This isolates the representation fix.
2. `s3-v-t1-policy.toml`: t1 + policy priors (c_puct 1.0).
3. `s3-v-t2-policy.toml`: t2 + policy priors (c_puct 1.0). Skip if t2 was skipped.

Each: `./apps/value_play/run.sh $(.venv/bin/python rundecks/elite-v3/render.py configs/NAME.toml)`.
Then compare: `--baseline elite-v3-v-mcts --candidate elite-v3-v-t1-value --candidate elite-v3-v-t1-policy --candidate elite-v3-v-t2-policy`
→ `results/stage3a.md/.json`.

**Gate G3a (choose the model):** `best` = whichever of t1 / t2 has the higher overall (`all`) HP-eq in its
policy arm; tie or t2 skipped: `elite-v3-t1`. Record `best` and its policy weight (t1 0.05, t2 0.25).

4. c_puct sweep with `best`: render `s3-v-best-c05.toml` and `s3-v-best-c2.toml` with `--var best=<best>`; run
   both; compare them (baseline `elite-v3-v-mcts`) → `results/stage3b.md/.json`.

**Gate G3b (choose c_puct):** `c_puct` = the value in {0.5, 1.0, 2.0} whose `best` arm has the highest overall
HP-eq (the 1.0 arm is the stage-3a run of `best`); ties → 1.0.

## Stage 4: one round of expert iteration (conditional, ≈80–100 min)

**Gate G4:** start only if elapsed ≤ T0 + 4h30. Otherwise skip to Stage 5.

1. Self-play: render `s4-selfplay.toml --var best=<best> --var c_puct=<c_puct>`, run with `./apps/fight_resample/run.sh`.
   The chosen net + policy search plays the other 2,436 training fights. Their root visits become policy targets.
   Then `check_data.py runs/schema=combat_v3/date=*/id=elite-v3-selfplay/out --min-fights 1500 --max-skipped 0.35`;
   GATE FAILED → skip the rest of Stage 4.
2. Train: render `s4-t3.toml --var best=<best> --var policy_weight=<its weight>`, run with `./apps/value_train/run.sh`
   (fine-tunes `best` on teacher + self-play data).
3. Validate: render `s4-v-t3-policy.toml --var c_puct=<c_puct>`, run with `./apps/value_play/run.sh`; compare with
   baseline `elite-v3-v-mcts` → `results/stage4.md/.json`.

## Stage 5: final test on the reserved 1,500 fights (conditional, ≈60–75 min)

**Choose the candidate:** among every validation arm run above (t1-value, t1/t2-policy, the c_puct sweep arms,
t3-policy), the one with the highest overall HP-eq vs `elite-v3-v-mcts`. Picking the best of several arms on the
same 450 fights flatters it; that is why the final test exists.

**Gate G5:** run only if (a) elapsed ≤ T0 + 6h45, (b) the candidate's validation overall HP-eq ≥ 0, and (c) none
of its three per-elite HP-eq point estimates is below −1.0. Otherwise do not touch the final fights: log why and
go to Stage 6.

1. `./apps/value_play/run.sh rundecks/elite-v3/configs/s5-f-mcts.toml` (MCTS on the 1,500 final fights).
2. The candidate: a policy arm → render `s5-f-candidate-policy.toml --var final_model=<model id> --var c_puct=<its c>`;
   the value arm → render `s5-f-candidate-value.toml --var final_model=elite-v3-t1`. Run with value_play.
3. Compare `--baseline elite-v3-f-mcts --candidate elite-v3-f-candidate` → `results/final.md/.json`.

## Stage 6: report (≈15 min)

1. Make sure every comparison above is in `results/`. Add a summary line per run to LOG.md: id, fights, skipped
   (diverged), wall time, seconds/fight from its `summary.json`.
2. Message **nn-consultant** over the intercom: "elite-v3 done; results in rundecks/elite-v3/results; LOG.md
   updated". nn-consultant writes REPORT.md. If there is no reply within 20 minutes, fill in
   `REPORT.md` yourself (the template has placeholders) with the tables from `results/`, verbatim.

## Budget sketch (8 h)

| Stage | Plan | Cumulative |
|---|---:|---:|
| 0 preflight | 0:15 | 0:15 |
| 1 teacher data | 1:35–1:50 | 2:05 |
| 2 train t1, t2 (+ MCTS validation in parallel) | 0:40 | 2:45 |
| 3 validation arms (5 × ~12 min) | 1:05 | 3:50 |
| 4 self-play, t3, validation | 1:30 | 5:20 |
| 5 final test (2 × 1,500 fights) | 1:10 | 6:30 |
| 6 report | 0:15 | 6:45 |

Rates are estimates from earlier 20k runs (MCTS ≈ 8.7 s/fight per worker, value-net leaves ≈ 12.9); the v3 net
and policy search are untested at scale, so treat them as ±50%. The gates keep the deck inside 8 hours.

## Files

- `configs/`: every app config (templates render into `configs/rendered/`).
- `render.py`, `check_data.py`, `compare.py`: helpers (read-only except their outputs).
- `LOG.md`: execution log (orchestrator). `REPORT.md`: results and conclusions. `results/`: comparison tables.
