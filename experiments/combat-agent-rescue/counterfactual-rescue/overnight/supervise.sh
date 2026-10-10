#!/bin/bash
# From repo root. rc0 finished -> exit0; rc2 Halt (research/SRE review) -> exit2, NO restart; rc124/137 hard deadline -> exit4;
# rc3 unexpected crash -> resume (max 3) ONLY if `run.py verify` (ledger/state/frozen recovery check) passes.
cd /home/sborowsk/project/sts_combat_rl
R=experiments/combat-agent-rescue/counterfactual-rescue/overnight/run.py
L=runs/schema=combat_v4/date=2026-10-06/id=overnight-rollout-v1/logs; mkdir -p $L
DEADLINE=$(date -u -d '2026-10-07 06:15:00 UTC' +%s)   # = 07:15 BST
tries=0
while true; do
  left=$(( DEADLINE - $(date -u +%s) )); [ $left -le 60 ] && { echo "$(date -Is) deadline reached"; exit 4; }
  echo "$(date -Is) SUPERVISOR start try=$tries"
  timeout -s TERM -k 60 $left .venv/bin/python $R run >> $L/run.log 2>&1; rc=$?
  echo "$(date -Is) SUPERVISOR rc=$rc"
  case $rc in 0) exit 0;; 2) exit 2;; 124|137) exit 4;; esac
  tries=$((tries+1)); [ $tries -gt 3 ] && exit 5
  .venv/bin/python $R verify >> $L/run.log 2>&1 || { echo "$(date -Is) verify failed; no resume"; exit 2; }
  sleep 30
done
