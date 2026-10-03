# Log (UTC)

## 23:20 — start, phase 0
- sts_lightspeed: committed user's rollout change (4ed3e57, enumerate legal actions only when the guided move is not
  taken; same moves/draws). HEAD already has the Falling OOB fix + Secret Portal disabled (a2c2e39).
- Frozen build `build/overnight/` (repo working tree incl. uncommitted combat_v4 recording from schema-v4 + my
  changes); worker copy `build/overnight/frozen/run_rl_worker` sha256 320ce3c2…; CTest 27/27 pass.
- Changes: `full` target (core.py, policy.terminal, learn/play CLI); `--key-rule` (ruby forced at Act 3 y=14 campfire
  even when rests are policy-decided; emerald routing in Policy.decide for Acts 2–3); sapphire taken only in Act 3
  (was: first chest of the run, which cost an Act 1 chest relic). impl: tests (test_full_target.py), analyze.py.
- Smoke (30 runs, incumbent act2-r02-train + rule, eps .05/route .15, combat_v4 recording on): no crashes, combat
  records written and replay-verified in-worker. 24 worker-s/run (~1200–1500 runs/h on 10 workers). Acts cleared
  {0:6,1:19,2:4,3:1}; one run entered Act 4 and reached the Heart (lost at 34 HP). Keys: emerald 43% (incl. Act 1
  burning elites taken by the elite-hungry net), ruby/sapphire only for runs reaching Act 3.
- v3 cost: forward ~9 ms/call on CPU, flat in batch size (GPU slower: ~15 ms/call, launch-bound). ~48 policy
  calls/run → ~0.5 s/run serialized in the play process: not a bottleneck at 10 workers.
- v3.1 training on smoke data works on GPU (--target full); untrained-ish v3.1 plays badly as expected.

## 01:05 — ah-c01 (1998/2000 runs) + fixes
- ah-c01 (incumbent act2-r02-train v1 + key rule [emerald routing Acts 2–3], eps .05, route .15): 79 min, 21.3
  worker-s/run. Full score 0.197 ± 0.002; act1 79.6%, act2 12.9%, act3 0.3% (6), Act 4 0.3%, **Heart kills 2/1998**
  (Heart fights 5, won 2). Keys: emerald 37%, ruby/sapphire ~10% (= runs reaching those points).
- Act 3: donu_and_deca win 14% (n=73, entry HP 39), time_eater 28% (58), awakened_one 40% (63). Entry HP 39–47.
- Act 2 clear 12.9% vs 19.3% in act2b-r03-collect (same eps/route, no Heart mode). Act 2 deaths dominated by
  elites (slavers 213, gremlin leader 192, book of stabbing 179 of 1998 runs). Attribution: emerald routing forced
  runs through Act 2 burning elites. Decision: emerald routing in **Act 3 only** (policy.emerald_options).
- Job failed at the end on 2 seeds: "combat_v4 rebuild does not match the played fight" (deterministic, 2 of ~40k
  fights). Worker now drops that fight's record with a stderr line instead of failing the run (worker sha bc5f500f).
  Accepted c01 as 1998 runs. **For schema-v4 owner: replay mismatch exists (seeds 971000000292, 971000001879).**
- ah-c02 relaunched (launch1b.sh): --key-rule-p 0.5 (rule on half the runs by seed → key-variation data for v3),
  then ah-dev-inc (rule on, greedy, 600 dev seeds).
- r0a training on c01 (GPU, MC lam 1): v1 init r02 val BCE 0.5217, v1 scratch 0.5223; v3.1/v3.2 running.
  v3 training: 9.7 GB RSS on 2000 runs (16 GB machine) and ~3 min/epoch at batch 32 → impl making learn.py lean.

## 03:00 — r0a offline screen (train c01 = 1998 runs, held-out 195 runs / 8488 nodes, MC full target)
| model | val BCE | R² all | R² act1 | R² act2 | R² act-start |
|---|---:|---:|---:|---:|---:|
| v1 init r02 | 0.5217 | 0.399 | 0.136 | 0.334 | 0.272 |
| v1 scratch | 0.5223 | 0.385 | 0.099 | 0.344 | 0.295 |
| v3.1 | 0.5229 | 0.362 | 0.111 | 0.242 | 0.278 |
| v3.2 | 0.5226 | 0.375 | 0.113 | 0.280 | 0.286 |
- No topology separates at 2k runs (differences ≪ noise of a 195-run held-out set; act-3 nodes n=481 from ~20 runs,
  R² meaningless). Same picture as Act 2 (data-limited). Next: 4k runs (c01+c02) and online paired gates.
- impl made learn.py streaming/compact: peak RSS 9.7 GB → ~1.4 GB (1900 runs); identical val batches.

## 04:15 — ah-c02, dev baseline
- ah-c02 (key rule on 50% of runs, emerald routing Act 3 only): 2000 runs, 0 failures/mismatches. Full 0.209 ± 0.002,
  act1 79.2%, **act2 17.4% (c01 12.9%: confirms the Act 2 emerald-forcing cost)**, act3 0.3%, Heart 0. 27 worker-s/run.
- **ah-dev-inc** (incumbent v1 + key rule, greedy, 600 dev seeds): full 0.233 ± 0.004, act1 85.8%, act2 23.7%,
  **act3 0.3% (2/600)**, Heart 0/600 (1 Heart fight, entered at 2 HP). 62 worker-s/run (CPU shared with GPU trainings).
- **Act 3 is the wall.** Of 142 Act-2 clears: died to Act 3 elites 47, hard hallway fights 46, bosses 44, cleared 2(+3?).
  HP entering Act 3's first fight 64.6. Act-3 boss win rates (dev+c01): donu_and_deca 14–16%, time_eater 23–28%,
  awakened_one 33–40%, and A20 needs two in a row. The v1 net has had no Act 3 training data (elite-hungry in Act 3).
- Process: orch/impl sessions dropped off intercom ~03:50 for a while; orch's own chain launched launch2.sh (I had
  changed it to evaluate r0b-v1td).

## 05:00 — r0b-v1td dev gate: better Act 3, worse Acts 1–2
- r0b-v1td = v1, TD(.7) continuing r0a-v1init (MC from the Act-2 incumbent on c01) on c01+c02 (3998 runs).
  r0b offline: all four models val BCE 0.5404–0.5407 (no separation; v3.1 still training at epoch 9).
- Paired dev vs incumbent (n=599; one seed lost to a worker `bad_function_call`, seed 961000000232):
  full −0.009 ± 0.005; act1 −2.7 ± 1.8 pp; **act2 −4.7 ± 1.9 pp**; **act3 +1.8 ± 0.6 pp (13 vs 2 clears)**;
  **Act 4 entered +1.7 ± 0.6 pp (12 vs 2)**; Heart 0/8 fights (entry HP ~56); floor −1.3 ± 0.5. Not promoted on
  the full score; but it's the first model that gets through Act 3 at all (Act-3 boss wins: donu 44% vs 16%,
  time eater 64% vs 23%, small n).
- Reading: the Act-2 incumbent (9k Act-2 runs of training) is better in Acts 1–2; the full-target model learned Act 3
  from ~800 Act-3 runs. Next: hybrid policy (incumbent for Acts 1–2, r0b-v1td from Act 3, `--late-ckpt/--late-act`)
  → dev gate, then collection ah-c03 with the hybrid (more Act-3 data). A crutch, like the key rule; the real fix is
  one model trained on enough data for all acts.
- Robustness: play.py now restarts the worker and skips a seed on worker errors (≤1% of seeds; listed in summary
  "seed_errors") instead of failing the job. Simulator fault to report: `bad_function_call` on dev seed 961000000232
  (r0b-v1td policy); combat_v4 replay mismatch on seed 961000000037 fights 11–14 (once desynced, every later fight).

## 05:45 — v3.2 dev gate: much worse in play despite equal val loss
- ah-dev-r0b-v32 (v3.2 from scratch, MC on c01+c02, key rule) vs incumbent, 600 paired dev seeds: full
  −0.073 ± 0.006, **act1 56.5% vs 85.8% (−29 ± 2 pp)**, act2 10.3% vs 23.7%, act3 0.8% vs 0.3%. Not promoted.
  v3.1 finished (val 0.5406, same as everyone); not played (v3.2 result + agreement below make it pointless tonight).
- Diagnosis (`agreement.py`, card picks logged by the incumbent on 150 dev runs; only picks keep all options in the logs):
  | act | n | v1td | v1init | v3.1 | v3.2 | skip rate logged / v1td / v1init / v3.1 / v3.2 |
  |---|---:|---:|---:|---:|---:|---|
  | 1 | 1224 | 0.69 | 0.58 | 0.46 | 0.45 | 21% / 16% / 11% / 4% / 7% |
  | 2 | 758 | 0.58 | 0.49 | 0.48 | 0.49 | 18% / 24% / 24% / 28% / 41% |
  | 3 | 161 | 0.50 | 0.41 | 0.47 | 0.46 | 19% / 24% / 19% / 24% / 46% |
  The scratch v3 nets take nearly every Act 1 card (deck bloat) and skip much more later: run-level outcome noise
  dominates the loss, so equal val BCE hides poor *relative* option values. Val BCE is not a usable selection metric
  for overworld nets; play (or decision-level checks) is.
- Control running: v1 from scratch on the same 4k runs (ah-r0b-v1) → dev gate, to separate topology from
  "no prior training" (the v1 candidates inherit 9k+ runs of Act 1–2 training via --init).
- Control (agreement only, no play): v1 from scratch on the same 4k runs agrees 0.50 / 0.49 / 0.49 (acts 1/2/3), skip
  13% / 30% / 13%: between v3 (0.45) and the init-from-incumbent v1s (0.58–0.69). So most of the v3 deficit is
  "no prior training", and a smaller part (~5 pp on Act 1 picks) is topology/optimisation. Not played (CPU budget).
- New: dense distillation in learn.py (`--distill CKPT [--distill-weight 1]`): the teacher's values on every offered
  option + skip are extra BCE targets (only the chosen column gets the MC/TD target otherwise). r0c: v3.1 / v3.2 from
  scratch + distill from r0b-v1td + MC, on c01+c02 (GPU, running).

## 06:20 — hybrid gate: Act-3 gains of v1td come from Acts 1–2 choices
- ah-dev-hyb (incumbent for Acts 1–2, r0b-v1td from Act 3), 600 paired dev seeds vs incumbent: full +0.0004 ± 0.0009,
  Acts 1–2 identical (as designed), act3 +0.2 ± 0.3 pp (3 vs 2 clears). So swapping only the Act-3 decision maker
  does nothing; r0b-v1td's 13 Act-3 clears come from how it plays Acts 1–2 (deck/relics built for later acts, at a
  cost of −2.7 pp Act 1 and −4.7 pp Act 2 clears). The hybrid crutch is dropped.
- Selection note: on the `full` score, r0b-v1td ≈ incumbent (−0.009 ± 0.005), but on Act 4 entries — the only route
  to Heart kills — it is +1.7 ± 0.6 pp (12 vs 2 of 599). For the true goal it is the better policy, so collection
  continues with it: ah-c03 (hybrid) stopped at 167 runs (kept as partial data), **ah-c04** = 2000 runs with
  r0b-v1td (eps .05, route .15, key rule on half the runs).
- Observation for the target: the `full` weights make a 4.7 pp Act-2 clear loss (≈ −0.005 + floors) roughly cancel
  a 1.7 pp Act-4 gain. Kept as agreed; flagged for discussion (Heart-relevant metric = Act 4 entries).

## 06:55 — distilled v3 (r0c): better, still short of the teacher
- r0c (v3.x from scratch + dense distillation from r0b-v1td, weight 1, MC): early stopping on the MC val loss stopped
  both after epoch 1 (val 0.5402/0.5404). Card-pick agreement with the incumbent: v3.1d 0.51/0.53/0.45 (acts 1/2/3),
  v3.2d 0.51/0.53/0.44 (scratch v3: 0.45/0.48/0.47; teacher v1td 0.69/0.58/0.50). Skip rates still off (Act 1 9%,
  Act 3 55% vs logged 21%/19%).
- Fix: val loss now includes the distillation term (MC + w·distill), so early stopping tracks the full objective.
  r0d: same with weight 2, batch 128 (GPU was ~40% busy at 32), both topologies in parallel.

## 08:05 — ah-c04, r0d distilled v3
- ah-c04 (r0b-v1td, eps .05, route .15, key rule on 50%): 2000 runs, full 0.205 ± 0.002, act1 78.7%, act2 15.0%,
  act3 0.9% (18), Act 4 0.4%, Heart 0; heart_locked 0.5% (rule-off runs reaching the end of Act 3 without keys —
  the counterexamples v3 needs). 29.7 worker-s/run.
- r0d (early stop on MC + 2·distill; batch 128): card-pick agreement with the incumbent, acts 1/2/3:
  **v3.2d 0.61/0.57/0.50**, v3.1d 0.58/0.55/0.46 (scratch v3 0.45/0.48/0.47; teacher v1td 0.69/0.58/0.50;
  v1init 0.58/0.49/0.41). Skip rates near logged (v3.2d 13%/28%/22%). Distillation transfers most of the teacher's
  relative values; attention (v3.2) fits it better than pooling (v3.1). → dev gate of v3.2d (ah-dev-r0d-v32d).
- Round 1 training running: ah-r1-v1td (TD .7 from r0b-v1td on c01–c04, ~6.2k runs).

## 09:15 — dev gates: distilled v3.2 is the best candidate; round-1 v1 not promoted
Paired dev (600 seeds, key rule on, greedy). Δ vs incumbent ± 1 SE:
| model | full | act1 | act2 | act3 / Act 4 entered | notes |
|---|---:|---:|---:|---:|---|
| r0b-v1td (v1, TD) | −0.009 ± 0.005 | −2.7 ± 1.8 pp | −4.7 ± 1.9 | +1.8 ± 0.6 / +1.7 ± 0.6 | n=599 |
| **r0d-v32d (v3.2, distilled from r0b-v1td + MC)** | **+0.003 ± 0.005** | −0.2 ± 1.9 | −0.7 ± 2.1 | **+1.2 ± 0.5 / +1.2 ± 0.5** | 9 vs 2 Act-4 entries |
| r1-v1td (v1, TD, 6.2k runs) | −0.012 ± 0.005 | −4.7 ± 1.8 | −5.0 ± 1.9 | +0.8 ± 0.4 | worse than r0b-v1td (−0.004 ± 0.004) |
| r0b-v32 (v3.2 scratch) | −0.073 ± 0.006 | −29 ± 2 | −13 ± 2 | +0.5 ± 0.4 | |
| hybrid | +0.000 ± 0.001 | 0 | 0 | +0.2 ± 0.3 | |
- r0d-v32d vs its teacher r0b-v1td (n=599): full +0.011 ± 0.005, act1 +2.5 ± 1.9, act2 +4.0 ± 1.9, act3 −0.7 ± 0.8.
  The distilled student keeps the incumbent's Act 1–2 strength *and* most of the teacher's Act 3 gain. First v3 net
  that plays at least as well as v1. Heart: 0 kills in all dev plays (a handful of Heart fights per 600 runs).
- Combat_v4 replay mismatches are more frequent late-game (c04: 44 dropped records in ~45k fights; dev v32d: 52).
- Launched launch8.sh: fresh final (800 seeds 981e9+) r0d-v32d vs incumbent, then dev r1-v32d, then fresh r0b-v1td.
