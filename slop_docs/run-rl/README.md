# Run-level RL: one value network makes every out-of-combat decision

*Entry point. Updated 2026-10-01. Chronological log: [RUNBOOK.md](RUNBOOK.md).*

## Goal and principle
Strongest autonomous Ironclad A20 agent; current scope **Act 1**. MCTS plays the fights. A learned run value network
V decides everything outside combat, trained and judged **only on real games**. Simulator- or model-internal
numbers are diagnostics, never progress. (This replaced the surrogate plan in `slop_docs/card-policy-loop/`, whose
weakness was optimising inside a biased simulator.)

## Method
- **V(state)** = expected run score from a state under the current policy. Score: 1 = Act 1 cleared, death on floor
  f = 0.25·f/16. Network: `run_policy_v1` (small deep set over deck / relics / potions + scalars + map as enumerated
  routes); value head only.
- **Every decision = argmax of V over the options' after-states:**

| decision | after-state per option | how computed |
|---|---|---|
| card reward | deck + card, or unchanged (skip) | exact (deterministic) |
| rest site | rest / upgrade card X / lift | exact, on a game copy |
| path | this state with only the routes through the next node | built, no simulation |
| shop | leave / buy card, potion, relic / remove card X | cards, potions, removals exact on a copy; relics built |
| event, Neow (floor-0 event) | outcome is random | **sampled lookahead**: 8 copies with fresh randomness per option, played on greedily until the event resolves (horizon 0); mean V of the ends |

  Everything else (card-select screens outside rest/shop, Dead Adventurer, Match and Keep, potion use): SimpleAgent.
- **Training loop** (`apps/run_rl/loop.py`): play 2,000 real runs (10% random exploration on every decision) ->
  retrain V on TD(λ=0.7) targets over the last 8 batches (older weighted 0.85 per step) -> greedy eval on 500 fixed
  seeds vs SimpleAgent -> repeat. Fresh-seed paired confirmations for any claim.
- **No peeking.** The agent never sees hidden randomness: exact after-states only where the outcome is deterministic;
  lookahead samples get fresh RNG streams, a fresh seed and re-shuffled relic pools; events that pre-roll hidden
  state (Dead Adventurer, Match and Keep) are excluded. Not yet re-drawn in samples: the pre-generated hallway/elite
  encounter lists (irrelevant at horizon 0; required before search across fights).
- **Fights inside lookahead** go through a pluggable `FightModel` (worker.cpp); today only `mcts_fight_model`
  (MCTS plays the whole fight). A combat-outcome-model version only has to return a FightModel.

## Results (Act 1 clear rate, real games, paired fresh seeds unless noted)
| policy | clear | vs previous |
|---|---|---|
| SimpleAgent (all decisions) | 59.9% | |
| + V card picks (v1) | 74.6% | +14.7 ± 1.6 |
| + rest + path (v2) | 88.9% | ~+14 |
| + shop (v3) | 91.1% | +2.2 ± 1.1 |
| + events, Neow by lookahead, no retraining | ~90% | +0.2 ± 1.1 (V untrained on those states) |
| v4: 2 rounds with every decision | pending | `rundecks/run-rl-v4` |

Findings: rest and path each add ~7-9 points (the policy avoids elites, rests more, reaches the boss with ~71 vs
54 HP). Network size / attention / auxiliary targets make no measurable difference (data-bound;
[network-study.md](network-study.md)). Cost ~7 worker-seconds per run (~6,000 runs/hour on 11 workers).

## Open issues
1. **Act-1-only objective** trades away later strength (fewer elites = fewer relics; resting over upgrading). Needs
   a longer horizon or a score term before going further.
2. **Search** (lookahead horizon > 0): needs encounter lists re-drawn in samples; when to search ("uncertain", e.g.
   ensemble disagreement) and how to evaluate it are open.
3. Remaining SimpleAgent decisions: card-select screens from events, potions, the two excluded events.

## Files
- Code: `apps/run_rl/` — `worker.cpp` (C++ worker: real runs, decisions asked of Python, `Player`, `FightModel`,
  sampled lookahead), `play.py` (plays runs with a policy, logs every step), `train.py` (TD training), `loop.py`
  (the RL loop), `common.py` (encoding, targets), `export.py` (parquet), `netstudy.py` (offline network screen),
  `neow.py` (retired bandit). CMake target `run_rl_worker` (develop in `build/run_rl_dev`).
- Networks: `python/sts_combat_rl/topology/run_policy_v1.py` (in use), `run_policy_v2.py` (study variants).
- Data: `runs/schema=run_rl_v1/date=*/id=*` (each with run.json; launch scripts in `out/`). DuckDB views
  `run_rl_results`, `run_rl_picks` (`sts_combat_rl.query.connect()`).

| run id | what |
|---|---|
| 2026-09-30/v1 | loop v1, card picks (iters 0-18) |
| 2026-10-01/confirm-v1-iter17 | v1 fresh-seed confirmation |
| 2026-10-01/v2-rest-path | loop v2, + rest + path |
| 2026-10-01/v2-ablation-archtest | rest-only / path-only ablation; real-game network test |
| 2026-10-01/netstudy | offline network screens |
| 2026-10-01/v3-shop, v3-fresh | loop v3, + shop; fresh-seed check |
| 2026-10-01/event-h0-test | events / Neow by lookahead vs control |
| 2026-10-01/v4-all | loop v4, every decision (rundeck run-rl-v4) |

- Other docs here: [RUNBOOK.md](RUNBOOK.md) (log), [checkins.md](checkins.md) (v1 learning curve),
  [network-study.md](network-study.md), [neow.md](neow.md) + [neow-handoff.md](neow-handoff.md) (retired bandit,
  kept for the record).

## How to run
```
cmake --build build/run_rl_dev --target run_rl_worker          # then snapshot the binary for a run
PYTHONPATH=python .venv/bin/python apps/run_rl/loop.py --root runs/schema=run_rl_v1/date=<UTC>/id=<name>/out \
  --iters N --window 8 --decay 0.85 --decide rest path shop event neow --init <model.pt> --seed-offset <unused> \
  --worker <binary>
```
Ground rules: kill processes by PID (a `pkill -f` pattern in your own command line kills your shell); one CPU-heavy
job at a time; every run uses its own worker copy.
