# Outcome-model topology sweep runbook (Opus-owned)

## Question
Given the card-outcomes data (~331k synthetic fights plus natural data, `experiments/card-outcomes`), does model capacity or
structure improve (a) the card-effect prediction (take vs skip) and (b) outcome prediction, compared with the frozen
legacy topology `co1-w16-h32-d30`? Data growth plateaued for the legacy net at card-outcomes s3c.

## Fixed
- **Data:** training uses `data/through-s3c.txt`. Dev/early-stop rows come from `data/eval-supp.txt` (the fixed s1 eval
  plus the devsupp runs), identical to the card-outcomes supplementary tables, so legacy runs reproduce them.
- **Training:** augmented arm only (the natural-only reference is known, and the topology does not change the
  natural-data question). Loss, sampling, patience 8, max 60 epochs and 32,768 examples/epoch are all unchanged.
- **Held-out data:** held-out groups never enter agents.combat.value. `hs16` (below) is dev-only, since natural bucket-8 donors are
  dropped from agents.combat.value.

## Topologies
- Specs live in `models/combat_outcome/architectures/*.toml`; see the README there. A spec is frozen once trained (`FROZEN.tsv`,
  enforced by `train_marginals.py`).
- Kinds:
  - `combat_outcome_v1`: the legacy class.
  - `combat_outcome_v2`: encounter-conditioned card encoder, sum+mean deck pooling, residual head.
- Grid 1 (`configs/grid1.txt`): 6 specs × lr {.003, .001} × training seeds {0, 1, 2} = 36 runs.

## Evaluation
1. **Supplementary dev** (41 boss / 60 elite donor clusters), scored with `experiments/card-outcomes/diag/reliability.py`.
   The metric is dMSE vs zero effect, which is unbiased under label noise. Brier and HP MAE come from `report.json`.
2. **High-seed card-effect set `hs16`:** 16 fight seeds per pair on bucket-8 donors. Boss: 30 groups. Hard/elite: 60
   groups. It is generated alongside the sweep and scored with `apps/combat_transition/eval_marginals.py` on the saved
   checkpoints (no retraining). Its pair means are precise enough to measure effect error directly.
3. Differences are judged against the spread across training seeds. Repeated inspection of the dev sets is
   exploratory, not confirmation.

## Schedule (≤ 8h compute; aim ≈ 5h)
| Block | Work | Wake |
|---|---|---|
| 0 | Smoke checks, then launch (~5 min) | short check |
| 1 | hs16 generation (8 workers, ~2h) ‖ grid 1 (2 parallel GPU jobs, ~1.5–2h) | ~2h |
| 2 | Review. Follow-up grid on the best 1–2 specs: new kind or training settings (epoch size, lr, dropout) chosen from the block-1 evidence (~1.5–2h) | ~2h |
| 3 | Final evaluation on hs16 and supplementary dev; `REPORT.md` | end |

## Commands
```bash
R=experiments/topology-sweep
$R/sweep.sh PREFIX $R/configs/GRID.txt $R/data/through-s3c.txt $R/data/eval-supp.txt 2   # resumable; skips done runs
experiments/card-outcomes/generate.sh $R/data/hs16.txt $R/configs/hs16-hard-elite.toml $R/configs/hs16-boss.toml
PYTHONPATH=. .venv/bin/python apps/combat_transition/eval_marginals.py \
  combat_transition_v1/2026-09-29/act1-eval-mcts-a20-checked --marginals $(cat $R/data/hs16.txt) \
  --checkpoints runs/schema=combat_outcome_v1/date=*/id=tsweep-*/out/augmented.pt --out $R/hs16-eval
```
Logs: `logs/`. Decisions: `DECISIONS.md`. Runs: `runs/schema=combat_outcome_v1/date=*/id=tsweep-*`.
