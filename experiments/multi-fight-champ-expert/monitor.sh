#!/usr/bin/env bash
# Read-only view of the overnight job (overnight.sh). Run from anywhere:
#   watch -n 30 experiments/multi-fight-champ-expert/monitor.sh            (five-fight run)
#   TAG=ten watch -n 30 experiments/multi-fight-champ-expert/monitor.sh    (ten-fight run)
cd "$(dirname "$0")/../.."
TAG=${TAG:-five}
TRAIN=$(ls -d runs/schema=combat_v4/date=*/id=champ-$TAG-rollout-v1 2>/dev/null | tail -1)
TEST=$(ls -d runs/schema=combat_v4/date=*/id=champ-$TAG-test-v1 2>/dev/null | tail -1)
LOG=$(ls -t scratch/multi-fight-champ-expert/overnight-$([[ $TAG == five ]] || echo "$TAG-")[0-9]*.log 2>/dev/null | head -1)

echo "== overnight job  $(date '+%F %T')"
if [[ -n $LOG ]]; then
    grep "overnight:" "$LOG" | tail -4
else
    echo "not started (no scratch/multi-fight-champ-expert/overnight-*.log)"
fi
echo "games running: $(pgrep -fc 'pv_worker (play|teacher)')   " \
     "GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)   " \
     "disk free: $(df -h . | awk 'NR==2 {print $4}')"

if [[ -n $TRAIN ]]; then
    echo
    echo "== 1/2 training"
    .venv/bin/python -m apps.combat_expert_iteration.status --run "$TRAIN"
fi

if [[ -n $TEST ]]; then
    echo
    echo "== 2/2 test  ($(grep -o '"status": "[a-z]*"' "$TEST/run.json" | head -1))"
    .venv/bin/python - "$TEST" <<'EOF'
import json, sys, tomllib
from collections import Counter
from pathlib import Path
out = Path(sys.argv[1]) / "out"
cfg = tomllib.loads((out / "config.toml").read_text())
for a in cfg["agent"]:
    total = cfg["run"]["fights"] * len(a["decks"])
    try:
        rows = [json.loads(l) for l in (out / a["name"] / "results.jsonl").read_text().splitlines() if l.endswith("}")]
    except FileNotFoundError:
        print(f"  {a['name']:<6} waiting ({total} fights)")
        continue
    wins, n = Counter(), Counter()
    for r in rows:
        d = r["fight_id"].split(":")[1]
        n[d] += 1
        wins[d] += r["status"] == "completed" and r["fight"]["won"]
    bad = Counter(r["status"] for r in rows if r["status"] != "completed")
    per = "  ".join(f"{d} {wins[d]}/{n[d]}" for d in a["decks"] if n[d])
    print(f"  {a['name']:<6} {len(rows)}/{total} played  {per}" + (f"  PROBLEMS {dict(bad)}" if bad else ""))
EOF
    [[ -f $TEST/out/REPORT.md ]] && echo "  report: $TEST/out/REPORT.md"
fi

if [[ -n $LOG ]] && grep -q "FAILED" "$LOG"; then
    echo
    echo "!! FAILED: see $LOG and the run's logs/stderr.log"
    tail -5 "$LOG"
fi
