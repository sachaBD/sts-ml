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
1,000 new seeds (800000000000+). **Net 74.6% vs SimpleAgent 59.9%: +14.7 ± 1.6 points** (paired). Slime 63 -> 82,
Guardian 63 -> 74, Hexaghost 54 -> 68. Matches the reused-seed estimate (no selection optimism).

**2026-10-01: rest + path decisions** (worker `decide: [rest, path]`, built in `build/run_rl_dev`). Every option is an
exact after-state computed on a game copy (rest: rest / smith X per distinct deck / lift; path: this state with only
the routes starting at that node). With SimpleAgent answering, 6/6 seeds reproduce no-decide runs exactly.
Committed f285235 (card-pick loop); decide + v2 topology uncommitted yet.

**2026-10-01: v2 loop** (`run_rl_v1/2026-10-01/v2-rest-path`): cards + rest + path by V. Starts from v1 iter017 model,
window 8 (initially v1 iters 11-18 data), decay 0.85, eps 0.1 on every decision, fresh training seeds
(offset 1e7), same 500 eval seeds with its own SimpleAgent baseline (61.2%, identical to v1's).
- Iter 0 (v1 iter17 net now also making rest + path, retrained once on its 2,000 runs + v1 data): **87.2% vs 61.2%,
  +26.0 ± 2.5**; cards-only v1 was ~76.5%.
- How: elites per run 1.25 -> 0.58, fights per run 6.9 -> 8.8 (more hallway fights = more card rewards), rests 827 vs
  SimpleAgent's 181 at the same sites, HP entering the boss 54 -> 71.5, reached boss 90 -> 96%.
- **Caveat:** this is the Act-1-only objective at work. Skipping elites skips relics, which Act 2+ needs. Fine for the
  current target; revisit (e.g. score at end of Act 1 with a relic / deck-strength term, or play into Act 2) before
  trusting it as a full-run policy.
- Iter 1: **91.4%, +30.2 ± 2.3**. Iter 2: 91.4%. Iter 3: 88.4%.
- Infra: train.py now keeps encoded batches on the CPU (all-on-GPU overflowed 12 GB under WSL and crawled: 45 min for
  4 epochs; now ~3 min). Lost ~40 min to that. Again killed my own shell with a pgrep pattern: use launch scripts.

**2026-10-01: network screen 1** (`scratch/netstudy/screen1.*`, GPU): run_policy_v2 variants on the fixed v1 data
(38k runs, 421k train / 76k val nodes, Monte Carlo targets). Val BCE (constant predictor 0.5255), 1 seed:
v1-size w32/h64 drop .3 **0.4623** (66k params, best epoch 9); wide w64/h256x2 drop .1 0.4698 (epoch 1: overfits);
wide + deck attention 0.4644; wide no-map 0.4727. Reading: capacity is not the limit on ~38k noisy runs,
regularisation is; attention helps the wide net; the map input helps. (GPU OOMs fixed: memory cap, split path-score
layer, embedding_bag path sum; yield.sh pauses the screen while the loop trains.)

**network screen 2** (`scratch/netstudy/screen2.*`): regularised variants, 2 seeds: small, small+attn, small no-map,
mid drop .5, mid+attn drop .4. Result: all 0.464-0.467 (seed noise ~0.002) except no-map 0.4705. No detectable
gain from size / attention; data-bound. Report: `network-study.md`.

## Plan (agreed 2026-10-01)
1. Confirmation (above).
2. Network study: fixed-data offline screen (Monte Carlo targets, held-out seeds), then short real loops for the best
   2-3 on fresh seeds. Report: `network-study.md`.
3. More decisions with the same V: rest sites (rest vs upgrade X), then path (next node via remaining routes).
   Shops / events deferred. Search at the pick deferred until the user is back.
