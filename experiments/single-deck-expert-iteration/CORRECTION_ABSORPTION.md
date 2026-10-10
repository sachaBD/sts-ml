# Demon Form correction absorption — Phase 1 (read-only diagnostic)

Requested by single-fight-astra 2026-10-06. No gameplay, training, teacher collection or final seeds.
CPU inference on existing rows only (~30 s, 2.4 GB peak RSS).

Question: was the teacher correction absorbed at the corrected decision states, and does the
training contract explain the weak gameplay effect (correction 70 vs control 69 /100 monitor;
uptake 40 vs 39 /100 training starts)?

## Reproduce

```sh
PYTHONPATH=. .venv/bin/python experiments/single-deck-expert-iteration/test_correction_absorption.py   # 5 synthetic checks
PYTHONPATH=. .venv/bin/python experiments/single-deck-expert-iteration/correction_absorption.py \
  --out runs/schema=combat_v4/date=2026-10-06/id=demon-form-correction-absorption-v1/out
```

Outputs: run `out/results.json` (all tables, provenance, sampler checks), `out/states.jsonl`
(per-state, per-model metrics), `out/logs/run.log`, `out/RUNBOOK.md`.

## Provenance (observed)

- update15 = correction `frozen/model.pt` = source `iter015/model/model.pt`, sha256 `97059033…`.
  control `model.pt` `c87d9258…`, correction `baa6b27a…`; their `model.onnx` hashes match the uptake run config
  (`f02e01d9…`, `80ba0d5b…`; `.onnx.data` not hashed there). Worker `58c5c4ab…` in all runs.
- Trainer source (`agents/combat/pv/{train,data,model}.py`) is uncommitted, last modified 2026-10-05 23:03, before
  the correction training (10-06 09:48). No run-time source snapshot exists: **provenance of the code that ran is
  unverified**, but strongly corroborated: the current `Dataset` code reproduces the logged per-epoch sampled
  states, policy states and concentration-weight sums exactly (all 9 epochs: control, correction, original update 15),
  and each saved checkpoint reproduces one logged validation line to <3e-8.
- ONNX vs PyTorch parity (256 teacher states, each model): |Δvalue| ≤ 6e-5 win-points, |Δlogit| ≤ 2.4e-6.
  The native worker runs the same ONNX via ORT C++; C++ live-encoding parity with stored rows **not checked**.

## Actual training contract (traced in code, verified by reconstruction)

- Train rows: 10 shards `iter006–015/rows` (+ `teacher-rows` for correction), filtered by `train-fights.json`;
  validation = `monitor-rows` (MCTS20k teacher trajectories on the 100 monitor seeds; 3,734 states).
- Sampling: 64 states per fight per epoch, with replacement, `np.random.default_rng(0)`; mixed-shard shuffle;
  1,000 batches of 64 per epoch. Model-independent RNG stream ⇒ exact replay.
- Policy target = normalized root visits; teacher-played action = argmax visits in 2,997/2,997 teacher policy states.
- Policy loss per batch = Σ conf·CE / (#policy states in batch). Not divided by Σconf: low-concentration states
  lose weight absolutely. conf = (n_legal·max p − 1)/(n_legal − 1). Value: MSE(100·won)/100².
- Checkpoint: best (value + policy) validation loss over 3 epochs. Only that checkpoint saved; **per-epoch
  checkpoints unavailable**. Selected: update15 **epoch 0**, control **epoch 2**, correction **epoch 0** (zero-indexed,
  confirmed by recomputation). So deployed control has 3 epochs of added training, correction 1 epoch.
  Correction epoch 0 vs 2 val score 0.707 vs 0.720, difference mostly value loss (0.216 vs 0.228).
  In the original run, val score ranges up to 0.17 across the 3 epochs of an update, and selection chose epoch 0 in 4/15 updates;
  the between-epoch differences are mostly in the 100-fight value loss. Observed fluctuation, not established
  stochastic label noise.
- Additional caveat: validation trajectories are on the same 100 monitor seeds later used to score the models.

## Exposure (exact reconstruction; "mass" = Σ conf/batch-policy-states, ≈420 per epoch total)

| Arm (deployed epochs) | Source | rows | samples | policy mass | mean conf (policy rows) |
|---|---|---:|---:|---:|---:|
| correction (ep0 only) | teacher replacement (100 fights) | 3,297 | 6,400 | 31.5 (7.5%) | 0.32 |
| correction (ep0 only) | retained learner (900) | 29,847 | 57,600 | 387.3 | 0.43 |
| control (ep0–2) | replaced learner losses (100) | 2,954 | 19,200 | 137.1 (10.6%) | 0.46 |
| control (ep0–2) | retained learner (900) | 29,847 | 172,800 | 1,160.4 | 0.43 |

Teacher states get 10% of samples but 7.5% of policy mass (teacher visits are flatter: mean max-visit share 0.46
vs 0.54 learner; Headbutt selections conf 0.15 vs 0.40). In the deployed correction model, each teacher state was
sampled ~1.9 times on average. Per stratum (deployed correction): Dual Wield selection 96 states, 187 samples,
mass 1.66 vs retained-learner DW mass 20.5; setup-legal 1,059 states, mass 10.9 vs 146.7.

Dual Wield soft-target pull (mass × target probability) in deployed correction: teacher Clash+ 0.84, Anger+ 0.37,
Boomerang+ 0.17; retained learner Clash+ 9.18, Boomerang+ 5.77, Bite 2.12, Bash+ 1.47, Anger+ 1.14.
Learner search targets already favour Clash+ over Boomerang+ in replay; the monitor play split (46/46) is not
explained by the targets alone (observation, not tested).

## Matched states: 100 teacher replacement trajectories (2,997 policy states, 100 fights)

These are **training states of the correction arm** (in-sample fit), not held-out. Point estimates
decision-level; brackets = 95% fight-cluster bootstrap (2,000 reps, seed 20261006), conditional on the fixed
trained models (no training-seed variability). Fight-balanced estimates are similar (in results.json).

| Metric | update15 | control | correction | corr−ctrl | corr−u15 |
|---|---:|---:|---:|---:|---:|
| KL(teacher visits ‖ net), nats | 0.521 | 0.546 | 0.492 | −0.054 [−0.062,−0.046] | −0.028 [−0.036,−0.020] |
| conf-weighted CE | 0.441 | 0.442 | 0.428 | −0.014 | −0.013 |
| P(teacher argmax set) | 0.346 | 0.347 | 0.352 | +0.004 [+0.002,+0.007] | +0.005 [+0.002,+0.008] |
| top-1 in argmax set | 0.445 | 0.438 | 0.447 | +0.009 [−0.000,+0.018] | +0.001 [−0.011,+0.013] |
| logit margin of argmax set | −0.154 | −0.186 | −0.109 | +0.077 [+0.056,+0.096] | +0.044 |
| value sq. err (/100²) | 0.211 | 0.196 | 0.155 | −0.041 [−0.054,−0.029] | −0.056 |

Strata (corr − update15, decision level):
- Dual Wield selection (96 states / 91 fights): P(argmax) 0.480→0.507 (+0.028 [+0.013,+0.043]); top-1 0.615→0.677
  (+6 states); KL −0.033. Fit ceiling if exactly matched: mean teacher max share 0.646.
- Setup-legal plays (1,059/100): P(argmax) 0.330→0.344 (+0.014); top-1 0.410→0.408 (no change).
- Rescued 44 fights (1,496 states): P(argmax) +0.010, top-1 +0.008 [−0.008,+0.025], value err −0.037.
  Unrescued 56 (1,501): P(argmax) +0.000, top-1 −0.005, value err −0.074.
- Control vs update15 moves *away* from teacher on these states (KL +0.025): 3 more learner epochs.

Secondary (100 replaced learner-loss trajectories, 2,647 policy states; different physical states): each arm
fits its own training data — control beats correction on learner targets (KL −0.020) and value (−0.044).

## Interpretation

Observed: corrections entered training (exact sampler replay), at reduced weight (7.5% of policy mass, flatter
targets), for one deployed epoch (~2 samples/state). Measured in-sample absorption is real but small:
P(teacher-best action) +0.5 pp overall, +2.8 pp at Dual Wield selections, top-1 unchanged except DW (+6/96
states). The value head moved more than the policy head (sq. error −27% vs update15), consistent with replacing
0/100 wins with 44/100.

Hypothesis (not established; low relative policy mass is NOT a demonstrated cause): the null gameplay result follows mainly from dose — 1 selected epoch, low/flat
policy mass, against a prior built over 15 updates — rather than from a plumbing failure. The correction−control
checkpoint-epoch mismatch (1 vs 3) is a recipe confound; it does not show later correction epochs were better.
Not ruled out: capacity/representation limits for these distinctions; teacher-visit targets that are only weakly
informative (concentration ≠ correctness); search at 2k sims overriding priors.

Recommended category: **entering but low-weight/weak targets, measurably but weakly learned at the corrected
states, no detectable gameplay gain.** Still unresolved whether a stronger dose would transfer to play.

Limits: one training RNG; per-epoch checkpoints absent; strata defined by recorded task/card IDs (DW task=4,
Headbutt task=10; setup = Demon Form/Corruption/Spot Weakness/Dual Wield legal); "played class" metric uses
byte-identical action tokens (network-indistinguishable, not proven physically equivalent) — exact-action
metrics are primary.

## Addendum: tied logits / indistinguishable action tokens

Legal moves with byte-identical action tokens (e.g. duplicate Bites) always get identical logits, but teacher
visits often split across them: in 972/2,997 teacher policy rows such a group straddles the argmax-set boundary.
Exact top-1 on ~19% of teacher states was decided by an index tiebreak among tied max logits. Secondary class
metrics (summed target / summed probability per identical-token group; not physical equivalence), correction −
update15: top-1 −0.3 pp [−1.7,+1.0], P(argmax class) +0.3 pp [−0.1,+0.6] overall; Dual Wield +4.2 pp
[−1.0,+9.4] and +2.5 pp [+1.1,+4.0]; setup-legal +0.5 pp and +1.3 pp. Conclusion unchanged.
Unweighted KL floor from these splits (jointly with the single duplicate-input pair): 0.1247 nats on teacher
rows (correction model's KL 0.49); concentration-weighted floor 0.0535. Derivation and tests: CORRECTION_FIT.md.

## Phase 2

Completed deliberate-fit diagnostic: see CORRECTION_FIT.md (update15 can fit these targets substantially, arguing
against a gross capacity/plumbing failure; general capacity/representation limits are not ruled out).
