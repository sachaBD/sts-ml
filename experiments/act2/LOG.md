# Log (UTC)

## 07:40 — start
- Context: overnight joint run (experiments/joint-expert-iteration): only overworld r00 promoted; fresh 88.6% (709/800) with
  neural combat leaf; neural combat was −4.0 ± 1.5 pp vs rollout at start. Act-1 deaths 80% at boss.
- Decision: combat = guided-rollout MCTS (works on any encounter, no act-2 training needed). Act-1 rollout baseline
  with r00 launched by orch (act2-a1-rollout-baseline, 800 fresh seeds paired with joint-1001-fresh-final).
- Existing target (core.score): clear 1, death 0.25·floor/16 — already floor-based but capped; replaced for act 2 by
  `floors` target: 0.4·floor/33 + 0.2·[act1 boss] + 0.4·[act2 boss] (full act-2 clear = 1). Old target kept as default.
- Combat search terminal value (35+hp+4·potions)/(55+maxhp) is per-fight, not run-aware: it already values HP and
  potions after the act-1 boss, so no change needed there (dropped from engineering list).
- Exploration audit: eps 0.1 covered card/rest/path/shop/event/neow; path random only per node. In fresh-final 54% of
  runs fought 0 elites (mean 0.55/run). Added per-run random routes (route_p, seeded by run seed) + eps on boss relic.
  SimpleAgent-only decisions (potion pickups, chest, etc.) remain unexplored.
- Per-node boss: multi-act runs need the current act's boss per training node (run-level boss = final act's).
- impl-10-2: worker C++ (max_act, act-2 paths, boss_relic decide, issue #5, per-step boss, seed in msgs, step guard).
- Me: core/learn/policy/play (target, per-node boss, route exploration, skip_allowed), test_multi_act.py, driver.py.

## 07:58 — rollout combat beats neural leaf in Act 1 (fresh seeds)
- act2-a1-rollout-baseline vs joint-1001-fresh-final, same 800 fresh seeds, same overworld r00:
  rollout MCTS 91.9% (735/800) vs neural-leaf 88.6% (709/800); paired diff **+3.25 ± 1.07 pp** (1 SE, n=800).
  Also cheaper: 11.3 vs 18.0 worker-s/run. Confirms decision: rollout MCTS for all act-2 work.

## 08:10 — worker done, smoke tests
- impl-10-2: build/act2/run_rl_worker — max_act, cross-act flow (branch map detach), act-2 paths, boss_relic decide
  (fresh-random after-states, pickup follow-ups by SimpleAgent), Library OBTAIN via pick (skip_allowed false; the only
  OBTAIN producer in the sim — issue #5: Neow/event REWARDS already went through pick), per-step boss, state act/boss,
  seed in msgs, step guard. CTest 22/22 pass; max_act=1 output identical on 3 seeds.
- Smoke 1 (20 runs, real sims, r00 model, eps .1, route .25): act-2 deaths early (mean floor 20.5); act-2 "weak"
  hallway fights at 500 sims cost 30–40 HP. Decision: act ≥2 easy fights use the hard budget (5k).
- Smoke 2 (60 runs, same seeds first 20): mean floor 22.9 (first-20: 22.95 vs 20.5), acts cleared {0:13, 1:41, 2:6}
  (10% act-2 clears, n=60, very noisy), 16.3 worker-s/run. Elites/run now 0–4 (25% 0) vs 54% 0 before.
- Cost ≈ 16 worker-s/run → 3000-run collection ≈ 80 min on 10 workers; 600-seed eval ≈ 16 min.

## 08:15 — controller launched; act-3 prep
- 08:07 orch launched experiments/act2/launch.sh (act2-controller): 3000-run collections, 600-seed paired dev gates,
  800-seed fresh final; init overworld = joint-1001-overworld-r00 (act1 target, run_policy_v1 width 32).
- impl: apps/run_rl/analyze.py (+test). Act-3 readiness (not used tonight): worker supports max_act 3, A20 double
  boss handled (first act-3 boss win does not end the run). Smoke at 50 sims, SimpleAgent: no crashes.
  **Simulator gap (sts_lightspeed, not fixed):** Falling event setup indexes counts[3] by CardType — CURSE/STATUS
  write out of bounds (A20 Ascender's Bane guarantees a curse); no bottled-card exclusion. Secret Portal advances
  floor only once (floor-based score understates). Filter Falling before act-3 production runs.

## 10:30 — round 0 gate
- Collection act2-r00-collect: 3000 runs took ~89 min (~18 worker-s/run, exploration runs longer).
- Dev gate (600 paired seeds, greedy): candidate act2-r00-train (floors target, MC λ=1, init r00) vs r00:
  score −0.001 ± 0.010; **act 1 clear 85.8% vs 93.7% (−7.8 ± 1.6 pp)**; act 2 clear 13.5% vs 10.7% (+2.8 ± 1.7 pp);
  mean floor 26.3 vs 26.1. Promoted by the first-round rule (target alignment; not clearly worse on score).
- Behaviour: candidate fights far more elites (runs with 0 elites 218→123; ≥3 elites 21→68). Deaths shift into
  act 1 (Hexaghost 18% of 200 boss fights; elites). Under the floors target the trade is score-neutral. It's the
  opposite of the "short-term power" worry: it buys long-term power (relics) at act-1 risk.
- Act 2 wall = bosses: death rates Automaton 72–77%, Collector 62–71%, Champ 56–64% (n≈60–90 each), entry HP ~57–62.
  Also Shelled Parasite+Fungi 20–26%, Slavers 33–42%, Book of Stabbing 30–33%.
- Offline value-net screen on r00-collect (val = 10% of seeds, MC floors targets): R² of V vs run score:
  act2-r00-train 0.20 (act1 nodes 0.09, act2 0.32); v1 lr 3e-4 0.22; v1 from scratch 0.23 (act1 0.10, act2 0.39).
  v2 val BCE no better (0.6804 vs 0.6775 v1); v2 checkpoint reload is broken (state-dict key mismatch) — not used.
  Outcome noise dominates; the early-stop at epoch 1–2 says data-limited. More data/TD is the lever, not capacity.
- Exploration cost (r00-collect, 3000 runs): random-route runs (n=750) score 0.418 ± 0.008, act1 70%, act2 4.5%,
  2.18 elites/run; policy-route runs (n=2250) 0.501 ± 0.004, act1 88%, act2 5.8%, 1.05 elites. Greedy dev play of
  the same model: act2 10.7%. Per-decision eps 0.1 over ~35 decisions/run is heavy; MC targets inherit that bias
  (TD λ=0.7 from round 1 partly corrects). Next session: eps ~0.05, route_p ~0.15.

## 12:35 — round 1 gate: real progress
- act2-r01-train (TD λ=0.7 bootstrapped from act2-r00-train, data r00+r01 collections, 6000 runs) vs act2-r00-train,
  600 paired dev seeds: **score +0.022 ± 0.011, mean floor +1.11 ± 0.31**, act2 clear 16.2% vs 13.5%
  (+2.7 ± 1.8 pp), act1 84.8% vs 85.8% (−1.0 ± 1.6 pp). Promoted.
- Cumulative vs the starting model (same dev seeds, not paired-tested directly): act2 10.7% → 16.2%, act1 93.7% → 84.8%.

## 14:45 — round 2 gate
- act2-r02-train vs act2-r01-train (600 paired dev seeds): score +0.026 ± 0.011, act2 clear 21.8% vs 16.2%
  (**+5.7 ± 1.7 pp**), act1 85.2% vs 84.8% (+0.3 ± 1.6), floor +0.2 ± 0.3. Promoted.
- Dev trajectory (act2 clear): r00 10.7% → 13.5% → 16.2% → 21.8%; act1 93.7% → 85.8% → 84.8% → 85.2%.
- Plan: fresh 800-seed final (selected vs r00) running; then continuation controller (launch2.sh: one more round,
  eps 0.05 / route_p 0.15, window r01–r03, dev evals reused by model name).

## 15:45 — fresh final + diagnosis
- FRESH (800 seeds 980000000000+, never trained/selected): act2-r02-train vs r00: act2 21.0% vs 10.75%
  (+10.25 ± 1.69 pp), act1 83.9% vs 91.6% (−7.75 ± 1.47 pp), score +0.048 ± 0.010, floor +1.89 ± 0.28.
- Act-1 regression diagnosis: HP entering act-1 boss 69.2 → 58.7. Floor-15 campfire: smith 704/776 (initial
  526/777). Smithing at 0–40% HP: 49 runs, boss win ~45% (vs 94% at 80–100% HP). Value net too coarse on HP vs
  upgrade. Decision: fix by search (rest lookahead to the next fight, MCTS on sampled futures) rather than a rule;
  impl-10-2 implementing behind a flag (play.py --rest-lookahead K,S). A/B next.
- Continuation controller (launch2.sh) started 15:41 by orch.

## 16:00 — rest lookahead implemented (not yet evaluated)
- Worker `rest_lookahead` {samples, sims_scale} (off by default; off = byte-identical output on 3 seeds). At a greedy
  net rest decision: compare rest vs the best non-rest option by k sampled plays (common random numbers, fresh
  randomness, map detached) through the next fight (+ rewards/boss relic) scored by V/terminals. Exploration
  choices are never refined; Policy.evaluate does not re-explore. play.py --rest-lookahead K,S.
- Cost (impl, n=5 seeds, SimpleAgent + synthetic V, serial under load): +40 ± 7 s/run (14.5 → 54.9 s), 4–9
  refinements/run. Too expensive to switch on blind: A/B first (600 dev seeds ≈ 55 min on 10 workers), and consider
  samples 2 or refining only when the next fight is a boss/elite.
- v2 reload fixed by impl (legacy path-scorer layout migration; all 27 v2 checkpoints under runs/ reload exactly).
  v2 from scratch on r00-collect: R² 0.18–0.19 vs v1 0.22–0.23 → no capacity gain at 3000 runs; v1 stays.

## 17:57 — round 3 (continuation) flat
- act2b-r03-train vs act2-r02-train (600 dev): score +0.002 ± 0.011, act2 +0.2 ± 1.8 pp, act1 +0.5 ± 1.4 pp → kept r02.
  Collection at eps .05/route .15: act2 19.3% (vs ~6% at eps .1/route .25), act1 79.8%.
- Plateau after 3 promotions. Launched rest-lookahead A/B (act2-dev-r02-restla, 600 dev seeds, 4 samples, sims ×0.25).

## 19:00 — rest-lookahead A/B
- act2-dev-r02-restla vs act2-dev-act2-r02-train (600 paired dev seeds): act1 +2.83 ± 1.09 pp (88.0 vs 85.2),
  act2 −0.67 ± 1.18 pp, score +0.007 ± 0.007, floor 27.97 vs 27.6; act-1 boss entry HP 60.3 vs 58.7.
  Worker time 55.4 vs 21.7 s/run (2.56×). Verdict: directionally right on act 1, not worth the cost as is.

## 22:30 — fight-outcome GUI covers Act 2 (incl. bosses)
- `apps/combat_transition/from_overworld.py` → `combat_transition_v1/2026-10-02/act2-overworld-fights`: 229k natural
  fights from all act2 overworld runs; pre-fight state rebuilt from the last recorded state (verified: HP = hp_before,
  same deck/relic ids as the post state) → 206k ok, 23k stale (dropped). Act 2 boss fights ok: champ 1981,
  automaton 1909, collector 2056. Final HP = post HP − Burning/Black Blood/Meat heal (9% capped → midpoint).
- `experiments/act2/train_fight_outcome.sh`: 3 seeds × {natural, augmented}, topology co2-w32-h64-l1-d30, lr .001,
  pair_w 1 (same as the old GUI model). Natural arm wins on natural dev for every seed (boss Brier 0.087–0.089 vs
  0.090–0.092; hexaghost pred 0.84–0.86 vs obs 0.836, augmented 0.76–0.80). Augmented is better calibrated on the
  Act 1 card experiments, which an older/weaker combat agent played. GUI uses natural.
- Ensemble on held-out act-2 boss fights (n=1766): Brier 0.149 vs base-rate 0.231. Per boss: automaton 0.131
  (n=559), champ 0.170 (n=594), collector 0.147 (n=613). Pred/real win: automaton 29.5/29.3%, champ 42.5/38.2%,
  collector 38.5/40.9%.
