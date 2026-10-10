# Overnight runbook: five-fight training + test (SRE)

One script, two stages, ~7.5 h total (rough, not measured; may run over). No decisions are needed while it runs.

## Launch (repo root, nothing else using pv_worker)

```sh
nohup experiments/multi-fight-champ-expert/overnight.sh > /dev/null 2>&1 &
```

The script prints and writes its log to `scratch/multi-fight-champ-expert/overnight-<UTC>.log`. It refuses to start if
other pv_worker games are running, or if either run id below already exists.

| stage | run (`runs/schema=combat_v4/date=<start date>/`) | expected |
|---|---|---|
| 1 train | `id=champ-five-rollout-v1` | bootstrap ~50 min (1,500 MCTS20k fights + 250 monitor), then 10 updates of ~25-40 min |
| 2 test | `id=champ-five-test-v1` | ~65 min: update-10 model and MCTS20k on 300 fresh seeds/deck; two-deck model on 2 decks |

## Watch

```sh
watch -n 30 experiments/multi-fight-champ-expert/monitor.sh      # everything below in one view
watch -n 30 '.venv/bin/python -m apps.combat_expert_iteration.status --run $(ls -d runs/schema=combat_v4/date=*/id=champ-five-rollout-v1)'
tail -f scratch/multi-fight-champ-expert/overnight-*.log
```

Healthy: the dashboard's "last write" stays under ~10 min (training steps and encoding write rarely, but well within
that), about 10 pv_worker processes during play stages, and the GPU in use during training.

## If something fails

- **Do not retry, re-launch, `--overwrite`, or delete anything.** The app has no resume, and a re-run would draw new
  seeds. Record what happened and stop.
- The script stops at the first failure and logs `FAILED (exit N) at line L`. Collect: the overnight log tail, the run's
  `logs/stderr.log`, the dashboard output, and `nvidia-smi` / `df -h` / `free -h`.
- The app exits by design if >5% of a play stage's fights error or time out (900 s per fight). Errors are never losses.
- If the test stage alone fails after training finished, report it. Rerunning only the test is the user's decision:
  `apps/combat_expert_iteration/evaluate.sh scratch/multi-fight-champ-expert/test_champ_five.toml`.
- Do not stop the job for slowness alone. Running past 8 h is fine.

## Report back

- Final status of both runs and total time.
- `champ-five-rollout-v1/out/REPORT.md` (monitor curve, 11 points per deck).
- `champ-five-test-v1/out/REPORT.md` (win rates and paired gaps per deck).
- Anything abnormal: errors/timeouts in the dashboard, warnings in stderr, restarts.

Interpretation belongs to the user/research agent. Monitor results are descriptive (the same 50 seeds reused each
update); the test stage is the result.
