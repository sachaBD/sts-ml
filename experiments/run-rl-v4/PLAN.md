# run-rl-v4: every decision by the run value network (2 rounds)

**Author: run-RL session (`subagent-chat-01a0f3f4`). Executor: orch-10-1.** Owner is around; keep it short
(~1.7 h). Log every launch / finish / problem in `LOG.md` (UTC times).

## Goal
v3 (cards, rest, path, shop decided by the run value network V; events and Neow by SimpleAgent) clears ~90-91% of
real Act 1 A20 runs (SimpleAgent ~60%). Events and Neow can now be decided by a sampled lookahead (copies of the game
with fresh randomness, V scores the outcomes), but V has barely seen the states those choices lead to, so on their
own they showed no gain yet (+0.2 ± 1.1). v4 = 2 rounds of gen -> train with every decision by V, so V learns those
states. Question: does v4 beat v3 on fresh seeds?

Background: `docs/research/run-rl/README.md`, `RUNBOOK.md` (log of everything so far).

## Design
- `apps/run_rl/loop.py`: per round, 2,000 real runs (MCTS fights, 10% random exploration on every decision) ->
  retrain V (TD(0.7), last 8 batches, older batches weighted 0.85 per step) -> greedy eval on the 500 eval seeds
  (600000000000+) against SimpleAgent (the loop plays its own baseline first).
- Start: v3 iter 3 model; training window starts with the last 4 v2 + 4 v3 batches. Fresh training seeds
  (offset 4e7).
- Worker binary frozen in `bin/run_rl_worker` (sha256 starts 2d2124d240ed445d; commit a9d38d9).
- Fresh-seed check: v4's final model on the 1,000 seeds already played by v3 in `event-h0-test` (830000000000+),
  paired.

## Ground rules (executor)
- Run from the repo root `/home/sborowsk/project/sts_combat_rl`. **Do not edit code, commit, stash or reset.**
- **One CPU-heavy job at a time** (each uses 11 workers on 12 cores). Stages run strictly in sequence.
- Never `pkill -f` with a pattern that also matches your own command line (it kills your shell); kill by PID.
- Interrupted? `loop.sh` and `fresh.sh` resume where they stopped (finished parts are skipped, partial play files
  continue). Just relaunch the same script. If a stage fails twice, stop and message the author.
- GPU: training uses it for ~5 min per round. Nothing else should be on the GPU.

## Stages
| Stage | Command | Est. | Check |
|---|---|---:|---|
| 0 preflight | `sha256sum experiments/run-rl-v4/bin/run_rl_worker` (starts `2d2124d240ed445d`); `nproc`; `free -g`; `nvidia-smi`; nothing else running (`ps -eo args \| grep run_rl`) | 2 min | record in LOG.md |
| 1 loop | `nohup experiments/run-rl-v4/loop.sh > /dev/null 2>&1 &` | ~80 min | `runs/schema=run_rl_v1/date=2026-10-01/id=v4-all/logs/stdout.log`; done when `out/curve.tsv` has iters 0 and 1 and the process exits |
| 2 fresh | `nohup experiments/run-rl-v4/fresh.sh > /dev/null 2>&1 &` | ~13 min | `.../id=v4-all/logs/fresh.log`; done when `out/fresh-iter001/summary.json` exists |
| 3 report | `PYTHONPATH=. .venv/bin/python experiments/run-rl-v4/report.py` | 1 min | writes `experiments/run-rl-v4/RESULTS.md`, parquet, run.json |

Log in LOG.md: start / end of each stage, the `CURVE` lines from stdout.log, the training-batch `clear_rate`s
(`grep clear_rate .../logs/stdout.log`), and the final RESULTS.md numbers. Then message the author (intercom
`subagent-chat-01a0f3f4`) with the RESULTS.md summary.

## What to expect
- Training batches clear less than eval (10% random choices). Eval was ~91-93% for v3 on the eval seeds.
- Per-run time ~7-8 worker-seconds; events add ~0.4 s. A round = ~25 min play + ~5 min train + ~6 min eval.
- Possible early artefacts (V on unseen floor-0 / event states), e.g. odd Neow picks: just log them.
