# Hard (hallway) fights: is MCTS good enough?

> **Status: DESIGN ONLY, NOT ACTIONED.** Shared rules, decision rule and X0: [../README.md](../README.md).
> Measurement rules: [../../evaluation_methodology.md](../../evaluation_methodology.md).

**Hypothesis (low confidence):** as with easy fights, no combat improvement on hard-pool fights is worth more than
0.5 act-clear points per run.

**Scope:** every regular (non-elite, non-boss) fight after a run's first 3 is drawn from the hard pool
(`category = 'hard'`). That is 1.6 per act-1 run on average, depending on the path; the first 3 are the easy pool.

**Why less sure than easy:**
- There are 10 encounters, some multi-enemy, with real HP costs: Gremlin Gang ~15 HP, Exordium Thugs ~13,
  Lots of Slimes ~10.
- Deaths are rare but non-zero (0.6%; Gremlin Gang and Thugs ~2%).
- They occur mid-act with built decks.
- The only prior comparison (single seed, value-net leaf, full-act evaluation) was level: −0.15 ± 0.30 HP,
  n = 286.

## Design

- **Fights:** 150 per encounter from buckets 0–1 of `act1-all-bosses-a20-scaled-search` (1,500 fights; ~1,400 after
  diverged replays). Hash-selected and frozen in `fights.csv`. Red Slaver has ~100–110 per bucket pair, so take
  what exists.
- **Arms:**

| arm | seeds | purpose |
|---|---|---|
| MCTS 20k | 0–3 | reference; noise floor; per-fight win probability |
| MCTS 2k, 5k, 50k | 0–1 | saturation (the full-act harness runs hard fights at 2k) |
| oracle 20k | 0 | ceiling; check convergence with oracle 50k on 300 fights |
| `ab-gen1` value net 20k | 0–1 | demonstrated fair headroom (trained on all categories, buckets 4, 6–9) |

- **Analysis:** as in easy/PLAN.md, per encounter and pooled.
  - Pooled: weight encounters by how often they occur per run (from `act1-eval-mcts-a20`), not equally.
  - Add the pivotal stratum (some fair run lost, the fight is winnable), scored on held-out seeds 4–5 if there are
    ≥ 20 such fights.

## Decision (pre-registered)

- **Stop** if ceiling impact = (oracle − MCTS) × 1.6 fights per run × carry-over 1 × X0 < 0.5 act-clear points.
- **Continue** (a v3 policy net for hard fights, t5 recipe) if the demonstrated gain × 1.6 × X0 ≥ 1 point, or the
  pivotal stratum shows a death-rate difference ≥ 1 point on any encounter.
- **Grey:** rank against macro work by cost.
- **Side result:** if MCTS 2k ≈ 20k, keep 2k in the harness. If 20k is clearly better than 2k, raising the
  harness budget is a free gain at act level.

## Cost

Hard fights are ~2–3 turns; assume ~1.5 s/fight/worker at 20k (pilot first).
~13 runs × ~1,400 fights on 11 workers ≈ **1.5–2 h**. The 50k arms are about half of that; drop them if MCTS is
already flat at 5k → 20k.

## Risks and caveats

- **Ten encounters × 150 fights leaves thin per-encounter intervals.** The decision uses the pooled estimate;
  per-encounter results only flag where to look.
- **The challenger is a value-only net.** On elites, value-only leaves lost to MCTS even though a policy-prior net
  later won. So "net ≈ MCTS" here does **not** mean there is no fair headroom. Only the ceiling can justify stopping.
