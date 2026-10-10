# Demon Form correction — Phase 2 deliberate-fit diagnostic

Approved by single-fight-astra 2026-10-06 (bounded; no gameplay, no promotion, no follow-on).
Question: can update15 fit the 2,997 teacher policy rows at all (capacity/plumbing)? This classifies fitting
ability only; it says nothing about gameplay benefit.

## Protocol (as run)

- Init update15 (`97059033…`); fresh AdamW lr 3e-4 wd 0.01 (not the saved moments: not a replication);
  grad-norm clip 1; batch 64; CPU 4 threads; torch seed 0. Full passes over all 2,997 teacher policy rows,
  per-epoch permutations from one `np.random.default_rng(0)`, shared by both arms (sha in `config.json`).
- Arms: **weighted** Σ conf·CE / n_policy (trainer objective) vs **unweighted** Σ CE / n_policy. Weighting changes
  total gradient scale as well as relative weights (clipped-step fraction logged: weighted 0–13%, unweighted
  9–36% of steps), so arm differences are not attributable to relative weighting alone.
- Policy loss only; no value loss, checkpoint selection, sharpening or argmax substitution. Checkpoints after
  epochs 0, 1, 3, 10, 30, all saved and all reported (no epoch picked).
- Validation = monitor rows: MCTS teacher policies on consumed monitor seeds, not a pristine held-out set.
- Run: 70.7 s total, within the 600 s cap; bg_wait exit 0. Metrics added afterwards by re-scoring the saved
  checkpoints (`--reevaluate`; the shared KL values reproduce exactly, Δ=0).

```sh
PYTHONPATH=. .venv/bin/python experiments/single-deck-expert-iteration/test_correction_overfit.py
PYTHONPATH=. timeout 660 .venv/bin/python experiments/single-deck-expert-iteration/correction_overfit.py --out $O --wall-seconds 600
PYTHONPATH=. .venv/bin/python experiments/single-deck-expert-iteration/correction_overfit.py --out $O --reevaluate
```
`$O = runs/schema=combat_v4/date=2026-10-06/id=demon-form-correction-fit-v1/out`; outputs `config.json`,
`{weighted,unweighted}/{train.jsonl,epoch*.pt}`, `results.json`, `results-eval.json`, `references.json`,
`logs/` (run logs + source snapshots used).

## Floors and references (derivation; tests in `test_correction_overfit.py`)

Model facts used (checked against `model.py` and by test): each legal move's logit is computed by the same
policy MLP from [shared trunk state features, that move's action token]; moves never interact. So moves with
byte-identical action tokens get identical logits regardless of position, and rows with byte-identical inputs get
identical outputs. No physical equivalence of moves is assumed.

- **Joint KL floor.** Minimize mean_i w_i·KL(p_i‖q) over achievable q. KL is convex and separates by input group;
  the optimum per input group is the w-weighted mean target, then averaged within each identical-token class
  (KL(p‖q) with q tied within class is minimized by the class mean of p). Verified by brute force in a test.
  Unweighted (w=1): **0.1247 nats**. Concentration-weighted (w=conf): **0.0535**. Of the unweighted floor, the
  whole-input duplicates alone contribute 7e-5 (1 pair of 2,997 rows; disjoint argmax sets); nearly all of it is
  within-row identical-token splitting (972 rows where such a class straddles the argmax-set boundary).
- **Argmax-set references** (train, all / Dual Wield / setup-legal):
  - exact-distribution reference, mass the teacher target puts on its own argmax set: 0.476 / 0.646 / 0.422 —
    not attainable where classes split, and a model may exceed it;
  - q* (class-averaged optimum) mass on argmax set: 0.420 / 0.646 / 0.390; q* top-1 (first-index tiebreak):
    0.797 / 0.969 / 0.822;
  - maximum achievable under tied class logits — top-1: 0.898 / 0.969 / 0.898; argmax-set mass: 0.822 / 0.965 / 0.857.

## Results (train = the 2,997 corrected states; endpoints and curve)

| Train, all states | ep0 | ep1 | ep3 | ep10 | ep30 |
|---|---:|---:|---:|---:|---:|
| KL weighted arm | 0.521 | 0.357 | 0.337 | 0.305 | 0.270 |
| KL unweighted arm (floor 0.125) | 0.521 | 0.317 | 0.292 | 0.260 | 0.217 |
| conf-weighted KL, weighted arm (floor 0.053) | 0.176 | 0.142 | 0.129 | 0.112 | 0.090 |
| top-1 exact, weighted / unweighted | .445/.445 | .474/.466 | .502/.494 | .524/.507 | .557/.540 |
| top-1 class, weighted / unweighted | .538/.538 | .599/.612 | .626/.649 | .640/.675 | .666/.728 |
| P(argmax set), weighted / unweighted | .346/.346 | .336/.301 | .348/.307 | .361/.321 | .385/.350 |

Fraction of reducible excess removed at epoch 30: unweighted KL (0.521−0.217)/(0.521−0.125) = 77%;
weighted arm, conf-weighted KL (0.176−0.090)/(0.176−0.053) = 70%. Still decreasing at epoch 30.
Unweighted P(argmax set) first falls (0.346→0.301): fitting flat targets lowers mass on the top move.

| Strata, train (weighted / unweighted) | ep0 | ep1 | ep3 | ep10 | ep30 |
|---|---:|---:|---:|---:|---:|
| Dual Wield (96) top-1 exact | .615 | .781/.719 | .802/.781 | .865/.823 | .854/.844 |
| Dual Wield P(argmax set) | .480 | .566/.471 | .609/.496 | .623/.533 | .653/.581 |
| Dual Wield KL | .390 | .300/.288 | .271/.232 | .212/.173 | .142/.091 |
| Setup-legal (1,059) top-1 exact | .410 | .434/.438 | .453/.468 | .471/.475 | .540/.524 |
| Setup-legal top-1 class | .442 | .549/.570 | .557/.601 | .583/.632 | .622/.703 |
| Setup-legal KL | .572 | .342/.294 | .324/.269 | .290/.232 | .248/.184 |
| Rescued (1,496) / unrescued (1,501) KL, weighted | .464/.577 | .365/.349 | .341/.333 | .312/.299 | .269/.271 |

Validation monitor rows (3,443; consumed teacher-policy validation): KL 0.477 → weighted .381/.371/.370/.394,
unweighted .366/.350/.338/.344 (ep1/3/10/30). Weighted rises after epoch 10. Dual Wield (105) top-1 .629 →
.771 (weighted ep30) / .762 (unweighted); setup-legal top-1 .439 → .443 / .471.

Value outputs (descriptive; shared trunk moves under policy-only loss): mean |Δvalue| vs update15 on train states
30–42 win-points overall (stratum means 23–54) across checkpoints. Train squared error rescued 0.248→0.03–0.08, unrescued 0.171→0.56–0.75:
value predictions moved upward broadly. Validation squared error 0.216→0.12–0.15. Not an isolated policy-head change.

## Classification (fitting ability only)

**Fits substantially, but not to the floor in 30 epochs.** Both objectives move the network far toward the teacher
targets at the corrected states, including Dual Wield selections (top-1 .615→.85; P(argmax) up to/above the
exact-distribution reference 0.646) and setup-legal states. One full teacher-only pass already reduces KL 0.52→0.32–0.36,
vs 0.52→0.49 for the deployed correction model. This argues against a gross capacity or target/loss plumbing failure
as the explanation for the weak Phase-1 uptake. It does NOT rule out capacity/representation limits in general
(e.g. for held-out states or finer distinctions) or remaining fitting limits (excess KL above the floor persists
at epoch 30). It does not identify which difference (dose, competing learner replay, fresh vs resumed optimizer,
checkpoint selection) matters, and does not imply gameplay benefit. Policy-only training also shifted value
outputs strongly, so any later use needs value control. No follow-on launched.

Reducible-excess percentages, each arm on each KL (floor-relative; (ep0−ep30)/(ep0−floor)):
unweighted arm: unweighted KL 77%, conf-weighted KL 63%; weighted arm: conf-weighted KL 70%, unweighted KL 63%.
