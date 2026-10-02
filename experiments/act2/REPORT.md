# Act 2 extension — report (2026-10-02)

Status: main controller done; continuation round (act2b) running. Plan: [PLAN.md](PLAN.md) · decisions: [LOG.md](LOG.md) ·
launches: [RUNBOOK.md](RUNBOOK.md) · controller tables: [CONTROLLER.md](CONTROLLER.md).

## Headline

The agent now plays Act 1 + Act 2 end to end. After three expert-iteration rounds the Act 2 clear rate roughly
doubled: **10.75% → 21.0%** on 800 fresh seeds. Act 1 got worse:

| Fresh seeds, n=800 paired (Ironclad A20, through the Act 2 boss) | Initial (r00, Act-1 model) | Selected (act2-r02) | Diff ± 1 SE |
|---|---:|---:|---:|
| Act 2 clear (both bosses) | 10.75% | 21.00% | **+10.25 ± 1.69 pp** |
| Act 1 clear | 91.6% | 83.9% | −7.75 ± 1.47 pp |
| Mean floor reached | 25.8 | 27.7 | +1.89 ± 0.28 |
| Floors score (training target) | 0.539 | 0.587 | +0.048 ± 0.010 |

These are seeds that were never used for training or model selection.

## What was built

- **Multi-act worker** (`apps/run_rl/worker.cpp`, `max_act`):
  - The run carries on through the act transition.
  - Path decisions work in every act.
  - The overworld agent chooses the boss relic. Each option is scored from an after-state built with fresh
    randomness; follow-up choices are made by SimpleAgent, the simulator's built-in rule-based agent.
  - The Library card choice goes through the card-pick decision (issue #5; it is the simulator's only
    obtain-a-card screen).
  - Every step records the current act's boss.
  - A step guard turns a hang into an error.
- **Act 3 ready but unused:** `max_act` 3 works, and A20's double final boss is handled.
- **Training target `floors`:** 0.4·floor/33 + 0.2·[Act 1 boss beaten] + 0.4·[Act 2 boss beaten]. A full Act 2
  clear scores 1.
- **Exploration:**
  - Each decision is random with probability ε. This now covers the boss relic, and runs can still reach elites.
  - A share of runs (by seed) take a uniformly random route.
- **Combat:** guided-rollout MCTS everywhere. Act 2 "easy" hallway fights get the hard-fight budget.
- **Tooling:** analysis tool `apps/run_rl/analyze.py` and the resumable controller `experiments/act2/driver.py`.

## Key findings

1. **Rollout MCTS beats the neural combat leaf in Act 1 and is cheaper.** Clear rate 91.9% vs 88.6% on 800 fresh
   paired seeds (+3.25 ± 1.07 pp). Time per run: 11 vs 18 worker-seconds.
2. **Each round improved Act 2** (600 dev seeds, paired against the previous model). Act 2 clear went
   10.7% → 13.5% → 16.2% → 21.8%. The score gains in rounds 1 and 2 were about 2 SE each.
3. **The Act 1 loss is mostly the value net misjudging HP versus card upgrades, not a deliberate trade:**
   - At the last campfire before the Act 1 boss, the selected model upgrades a card instead of resting 91% of the
     time; the initial model did so 68% of the time.
   - It upgrades even at 0–40% HP: 49 of 800 runs, which then beat the boss only about 45% of the time.
   - As a result, HP entering the Act 1 boss fell from 69.2 to 58.7.

   An offline check found the value net explains only about 20% of the variance in run outcome (R² ≈ 0.2), so it
   is too coarse to weigh this choice well. The fix is in progress: a lookahead for the rest decision that plays
   the next fight out with MCTS on sampled futures (feature off by default; not yet evaluated).
4. **Act 2 bosses are where runs now end.** Selected model, fresh seeds:

   | Boss | Death rate | n |
   |---|---:|---:|
   | Automaton | 69% | 153 |
   | Champ | 60% | 149 |
   | Collector | 56% | 138 |

   HP entering the Act 2 boss averages about 61.
5. **Exploration is expensive:**
   - Random-route runs reached Act 1 clear only 70% of the time, against 88% for policy routes.
   - With ε = 0.1 per decision, collection runs clear Act 2 about 5–6% of the time, against 11% when the same
     model plays greedily.
   - The continuation round therefore uses ε = 0.05 and random routes in 15% of runs.

## Caveats

- **The model was picked on dev seeds; only the final comparison is fresh.**
- **The value net is small** (run_policy_v1, width 32).
  - The larger v2 model scored no better on held-out data (val BCE), and its checkpoints don't reload (state-dict
    key mismatch).
  - Training stops early, after 1–2 epochs, which suggests data is the limit, not model size.
- **Simulator gaps (not fixed):** the Falling event (Act 3) writes out of bounds, and Secret Portal advances the
  floor counter only once.
- **SimpleAgent still makes some decisions,** such as potion pickups and card removal or transform screens in
  events. These are never explored.

## Next steps (recommended)

1. **Evaluate the rest lookahead.** Run an A/B on dev seeds; if it helps, extend the lookahead to other
   high-stakes choices such as boss relics and elite routes.
2. **Keep iterating rounds.** Each round costs about 2.3 h on 10 workers and has still been improving.
3. **Act 2 boss combat:**
   - Test larger boss search budgets.
   - Measure potion use in boss fights.
4. **Act 3:** filter the Falling event, then train with the `floors3` target.
