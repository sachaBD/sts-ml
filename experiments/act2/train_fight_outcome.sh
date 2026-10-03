#!/usr/bin/env bash
# Fight-outcome ensemble with Act 2 fights for the GUI (same topology/settings as the Act-1 GUI model tpair1-*).
# Natural rows: combat_transition_v1/2026-10-02/act2-overworld-fights; synthetic: the Act-1 card-marginal runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
NAT=combat_transition_v1/2026-10-02/act2-overworld-fights
M="card_marginals_v1/2026-09-29/card-outcomes-s1-easy card_marginals_v1/2026-09-29/card-outcomes-s1-hard-elite card_marginals_v1/2026-09-29/card-outcomes-s1-boss card_marginals_v1/2026-09-29/card-outcomes-s2-devsupp-boss card_marginals_v1/2026-09-29/card-outcomes-s2-devsupp-hard-elite card_marginals_v1/2026-09-29/card-outcomes-s2-easy card_marginals_v1/2026-09-29/card-outcomes-s2-hard-elite card_marginals_v1/2026-09-29/card-outcomes-s2-boss card_marginals_v1/2026-09-30/card-outcomes-s3-easy card_marginals_v1/2026-09-30/card-outcomes-s3-hard-elite card_marginals_v1/2026-09-30/card-outcomes-s3-boss card_marginals_v1/2026-09-30/card-outcomes-s3-boss-b card_marginals_v1/2026-09-30/card-outcomes-s3c-hard-elite card_marginals_v1/2026-09-30/card-outcomes-s3c-boss"
E="card_marginals_v1/2026-09-29/card-outcomes-s1-easy card_marginals_v1/2026-09-29/card-outcomes-s1-hard-elite card_marginals_v1/2026-09-29/card-outcomes-s1-boss card_marginals_v1/2026-09-29/card-outcomes-s2-devsupp-boss card_marginals_v1/2026-09-29/card-outcomes-s2-devsupp-hard-elite"
for s in 0 1 2; do
  INPUTS=""; for r in $NAT $M; do INPUTS="$INPUTS --input $r"; done
  CARD_OUTCOME_SEED=$s CARD_OUTCOME_LR=.001 CARD_OUTCOME_PAIR_W=1 PYTHONPATH=. .venv/bin/python -m runs.run combat_outcome_v1 \
    act2-co2-w32-h64-l1-d30-lr.001-s$s $INPUTS --note "act1+act2 natural + act1 marginals; seed=$s lr=.001 pair_w=1" -- \
    .venv/bin/python apps/combat_transition/train_marginals.py $NAT --marginals $M --eval-marginals $E --epochs 60 \
    --device cuda --out {out} --topology models/combat_outcome/architectures/co2-w32-h64-l1-d30.toml --arms natural augmented
done
