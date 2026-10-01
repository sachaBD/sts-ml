# Neow: a bandit (RETIRED)

> **RETIRED 2026-10-01.** Not pursued. The premise below ("everything else about the moment is fixed") is wrong: the
> map and the act boss are known at Neow and differ per run, so the choice is contextual and a per-arm bandit
> ignores that context. Neow will be handled as a regular decision scored by the value network over sampled
> outcomes (see `slop_docs/neow_handoff.md`). The bandit code (`apps/run_rl/neow.py`) is kept only so the existing
> wiring runs; kept here for the record.

## The problem
At the start of every run Neow offers 4 bonuses; we take one. Everything else about the moment is fixed: same
starting deck, HP and gold, no map choices made yet. The only thing that varies is which 4 bonuses are offered (and
the known act boss).

So this is not a planning problem. It is: "across many runs, which bonus leads to the most Act 1 clears, given what
we do afterwards?" That is a **multi-armed bandit**. Each kind of bonus is an arm; each run pulls one of the 4 arms
offered and pays out its final score.

## What the arms look like (sts_lightspeed)
Every run offers one arm from each of 4 slots, each drawn uniformly:
| slot | arms | examples |
|---|---|---|
| 1. card bonus | 6 | choose 1 of 3 cards, random rare card, remove / upgrade / transform a card, choose a colorless card |
| 2. small bonus | 5 | 3 potions, random common relic, +10% max HP, Neow's Lament (first 3 enemies at 1 HP), 100 gold |
| 3. bonus + drawback | 25 | e.g. remove two cards but lose all gold; rare relic but gain a curse; 250 gold but take 30% HP damage |
| 4. boss relic swap | 1 | random boss relic, lose Burning Blood |
37 arms in total. An arm = bonus + drawback (e.g. `remove_two|no_gold`). SimpleAgent always takes slot 1.

## Why a bandit, not the value network
- **Outcomes are random** (which rare card, which boss relic, which transform). Our value network scores exact
  after-states; computing those for Neow would mean peeking at the actual roll. The bandit only learns from what
  happened, so it cannot cheat.
- **No context needed.** The state is the same every time, so a table of per-arm results is all the "model" we need.
- **Simple and honest.** It cannot be fooled by value-network errors on unusual floor-0 states.

## Our solution: Thompson sampling
- Keep, per arm, how well runs that took it went (successes / failures of the run score).
- **Training runs:** for each offered arm, draw a plausible success rate from what we know about it, take the
  highest. Arms we know little about sometimes draw high, so they get tried (explore); arms known to be good win most
  draws (exploit). Exploration fades on its own as evidence accumulates.
- **Evaluation runs:** take the offered arm with the best average (no exploration).
- **The rest of the policy keeps improving**, so an arm's value drifts. Old evidence is gradually down-weighted so the
  table tracks the current agent.

## What to expect
- **Slow to resolve some arms.** A run offers one slot-3 arm out of 25, so each is offered in ~4% of runs (~80 per
  2,000-run batch), and only taken some of that time. Telling apart arms that differ by a few points at a ~90% clear
  rate needs on the order of 1,000 runs each. Slots 1, 2 and 4 resolve much faster.
- **Probably a small gain overall** (a few points at most), but cheap, and it removes one more fixed rule.
- **Act-1 bias applies here too.** E.g. a boss relic swap or 250 gold might be judged only by Act 1 clears.

## Possible refinements (later, only if useful)
- Share evidence across arms: a bonus's value and a drawback's cost are roughly additive, which would let rare
  slot-3 combinations borrow strength from the others.
- Per-boss tables (triples the data needed).
- Follow-up choices (which card to remove / upgrade, which of 3 cards to take) can use the value network, since
  those are exact, deterministic after-states. Card choices already do.

## Implementation spec (for the implementing agent)
- Interface and wiring are done: worker `decide: neow` (offers the 4 arms, no after-states), `apps/run_rl/neow.py`
  (`NeowPolicy` stub: always option 0), `play.py --neow`, `loop.py` (per-iteration `iterNNN/neow.json` = previous
  state + update on that iteration's training batch; eval uses greedy choose). Verified: with the stub, runs are
  identical to SimpleAgent's Neow choice (12/12 seeds).
- To implement in `NeowPolicy` only (keep the interface): Thompson sampling with Beta posteriors per arm over the
  run score (scores in [0, 1]: count score as success weight, 1 - score as failure weight), uninformative prior,
  discount of old evidence per update (e.g. 0.9 per batch, configurable), greedy = posterior mean when
  explore=False, deterministic RNG (seeded) for reproducibility, readable `summary()`.
- Test: offline, replay the per-arm counts from a logged batch; then a short loop (`--decide rest path shop neow`)
  and a paired fresh-seed comparison against the same policy without `neow`.
