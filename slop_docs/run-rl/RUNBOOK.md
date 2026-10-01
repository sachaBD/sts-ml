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
- Iter 1: **91.4%, +30.2 ± 2.3**. Iters 2-5: 91.4, 88.4, 90.8, 91.2. Plateau from iter 1 (window already 8 + decay);
  stopped after iter 5 to free compute.
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

**2026-10-01: ablation + real-game network test** (`run_rl_v1/2026-10-01/v2-ablation-archtest`, out/cmd.sh).
- Ablation, v2 iter5 model, 500 eval seeds (paired): cards + rest + path 91.2%; cards + rest only 84.2% (-7.0 ± 1.9);
  cards + path only 82.6% (-8.6 ± 1.6); cards only (v1 iter17) 77.6% (-13.6 ± 2.1); SimpleAgent 61.2%.
  Rest and path each add ~7-9 points and are roughly additive.
- Network test, 1,000 fresh seeds (810000000000+), cards + rest + path, all trained on the same v2 data (iters 0-5)
  with the same TD targets: loop model iter5 88.9%; small (v1 size) from scratch 89.4% (+0.5 ± 1.1); mid + aux 88.5%
  (-0.4 ± 1.2). **Tie**: confirms the offline screen. Fresh-seed level of v2 ≈ 89% (eval seeds 91%: mild optimism).

**2026-10-01: shop decisions** (worker `decide: [..., shop]`): options leave / buy card / buy potion (free slot) /
buy relic / remove card X, asked after every purchase until leave. Card / potion / removal after-states exact on a
copy; relic after-states are built (relic added, gold paid) because some relics roll on pickup: no peeking at hidden
outcomes. Events and Neow stay with SimpleAgent: their outcomes are random, so exact after-states would leak.
Fidelity: SimpleAgent answering, 12/12 seeds identical to no-decide runs.
Context (v2 iter5 eval): shop on the chosen path in 33% of runs (SimpleAgent 40%); SimpleAgent always removes
(usually a Defend), buys cards by a priority list and any affordable relic, never potions.

**2026-10-01: v3 loop** (`run_rl_v1/2026-10-01/v3-shop`, scratch/run_v3.sh): cards + rest + path + shop. Init v2 iter5,
window 8 (v2 iters 0-5 data first), decay 0.85, eps 0.1, training seeds offset 2e7, same 500 eval seeds.
- First training crashed (my aux-target code mistook run_policy_v1's untrained extra heads for aux outputs; fixed:
  aux only when the model has `aux_out`) and nearly OOMed in TD values (chunk 2048 x ~1,250 routes; now 256). ~45 min
  lost; data intact, resumed.
- Iter 0: 90.8% (+29.6 ± 2.4), same level as v2 (~90.5%). Shop behaviour after one batch: buys cards 329 / leaves
  116 / removes 110 / potions 95 / relics 2 (SimpleAgent: removes 327, leaves 175, cards 141). **Red flag: removes
  Bash 88 times**: possibly a value artefact with little shop data; watch whether it persists.
- Iter 1: 91.4%; Bash removals 88 -> 44, Defend 6 -> 17 (correcting with data). Iter 2: 92.0%. Iter 3: 93.0%
  (last; eval seeds, so slightly optimistic: fresh-seed level of iter 2 was 91.1%).
- **Fresh seeds** (`run_rl_v1/2026-10-01/v3-fresh`; the 1,000 seeds of the v2 network test): v3 iter2 91.1% vs v2
  iter5 88.9%, **+2.2 ± 1.1**. Same HP entering the boss (71) and reach rate (97%); spends more gold (36 left vs 49).
  Modest, ~2 SE. v3 also had 3 more training iterations, but v2 was already flat.
- Loop resumed for iter 3, then stopped by `scratch/stop_v3_after_iter3.sh` (session budget), which also exports
  parquet and writes run.json.

## Where things stand (end of 2026-10-01 session)
| policy (V decides; SimpleAgent the rest; MCTS fights) | Act 1 clear, fresh seeds |
|---|---|
| SimpleAgent | 59.9% (1,000 seeds) |
| cards | 74.6% |
| cards + rest + path | 88.9% |
| cards + rest + path + shop | 91.1% |
- Network size / attention / aux targets: no measurable effect (`network-study.md`). Data-bound.
- Biggest caveat: the objective is Act 1 only. The policy avoids elites (fewer relics) and rests a lot; that may hurt
  Act 2+. Before optimising further, extend the score or the horizon (e.g. play into Act 2, or add a relic / deck term).

## Suggested next steps
1. Horizon: decide how to stop the Act-1-only objective from trading away Act 2 strength (elites, relics, upgrades).
2. Search at the decision (deferred, with the user): rollouts or gauntlet on top of V for close calls.
3. Remaining decisions: events / Neow need sampled outcomes (no exact after-states without peeking); potions.
4. Housekeeping: launchers are ad-hoc scripts in scratch/; run.json files are hand-written.

**2026-10-01 (evening): Neow handed off; sampled lookahead built (events).**
- Neow: interface + stub + `neow.md` (bandit framing); implementation handed to the `neow` session.
- Worker refactor: one `Player` plays both the real game and lookahead samples; fights go through a pluggable
  `FightModel` (only `mcts_fight_model` = MCTS plays the whole fight; a combat-model version only needs to return a
  FightModel). Behaviour unchanged: 24/24 regression seeds identical (SimpleAgent and net policies).
- Sampled lookahead (`decide: event`, `lookahead_samples` 8, `lookahead_horizon` 0): per option, 8 copies with fresh
  randomness (same per sample index across options), option applied, played on greedily until back on the map
  (h = 0) or `h` floors later; Python averages V over the ends (death = its score, clear = 1). Inside samples,
  nested event steps use one-step after-states. Horizon > 0 = search (not yet used; the pre-generated encounter
  lists must be re-drawn in samples first, or search would see upcoming fights).
- No-peeking rule extended: Dead Adventurer and Match and Keep pre-roll hidden state at setup (reward order /
  elite; face-down board), which a copy would reveal: left to SimpleAgent.
- Cost: ~0-3 s per event decision, except Match and Keep (now excluded; ~70 s).
- Paired test running: `run_rl_v1/2026-10-01/event-h0-test` (v3 iter3 model, 1,000 fresh seeds 830000000000+).
