# run_rl runbook / notebook

Concise log of what was done, newest last. Design: `README.md`. v1 check-ins: `checkins.md`.
Data: schema `run_rl_v1` (`runs/schema=run_rl_v1/...`); DuckDB views `run_rl_results`, `run_rl_picks`
(`sts_combat_rl.query.connect()`; written by `apps/run_rl/export.py`).

## Ground rules
- Only real games count. Offline metrics only screen candidates.
- Reference = SimpleAgent picks on the same seeds, through the same worker binary.
- Every run uses its own copy of the worker binary (`--worker`; loop.py snapshots it). Develop in `build/run_rl_dev`.
- Kill processes by PID, never `pkill -f` with a pattern that appears in your own command line.

## Log
**2026-09-30/10-01: v1 loop** (`run_rl_v1/2026-09-30/v1`): card picks only, run_policy_v1 w32/h64, eps 0.1, TD(0.7),
reward clear=1 / death 0.25*floor/16. 500 eval seeds (600000000000+). SimpleAgent 61.2% -> net ~76.5% (iters 3-17
flat; window 3 -> 8 + decay 0.85 at iter 9 changed nothing visible). Stopped by agreement after iter 17.

**2026-10-01: housekeeping.** Moved v1 into the schema layout (hand-written run.json), smoke run to scratch/.
Added query views. play.py `--worker`, loop.py snapshots the worker.

**2026-10-01: fresh-seed confirmation** (`run_rl_v1/2026-10-01/confirm-v1-iter17`): iter 17 model vs SimpleAgent,
1,000 new seeds (800000000000+). Result: pending.

## Plan (agreed 2026-10-01)
1. Confirmation (above).
2. Network study: fixed-data offline screen (Monte Carlo targets, held-out seeds), then short real loops for the best
   2-3 on fresh seeds. Report: `network-study.md`.
3. More decisions with the same V: rest sites (rest vs upgrade X), then path (next node via remaining routes).
   Shops / events deferred. Search at the pick deferred until the user is back.
