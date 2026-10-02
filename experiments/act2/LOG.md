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
