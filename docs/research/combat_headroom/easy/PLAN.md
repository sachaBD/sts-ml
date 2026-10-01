# Easy fights: is MCTS good enough to stop?

> **Status: SUPERSEDED (2026-09-29).** The stop-rule framing below was dropped (it could only end in "grey, depends
> on X0"). What ran instead, a cheap-agent budget study on the same fights:
> [experiments/act-1-combat/2026-09-29-easy-cheap-agent/](../../../../experiments/act-1-combat/2026-09-29-easy-cheap-agent/README.md).
> Shared rules, decision rule and X0: [../README.md](../README.md).
> Measurement rules: [../../evaluation_methodology.md](../../evaluation_methodology.md).

**Hypothesis:** on easy-pool fights (Cultist, Jaw Worm, Two Louse, Small Slimes), no combat improvement is worth
more than 0.5 act-clear points per run. Stop researching them.

**Scope:** StS draws a run's **first 3 regular fights** from the easy pool and every later one from the hard pool
([../hard/PLAN.md](../hard/PLAN.md)). So a run has at most 3 easy fights (2.9 on average). The 1st is always
floor 1. The 2nd and 3rd are usually on floors 2–5, and later if the path avoids monster rooms.

**What's already known** (`experiments/act-1-easy-combats/`, floor 1 only, 100 fights per encounter):
- MCTS is flat from 1k to 100k simulations.
- The oracle ceiling is +3.3 HP/fight; a fair net gained +0.57.
- No deaths.

**Gap:** the 2nd and 3rd easy fights (about 65% of easy fights) were never measured. They come with Neow and
card-pick decks and less HP. This study covers them and converts the result to run impact.

## Design

- **Fights:** 250 per encounter from buckets 0–1 of `act1-all-bosses-a20-scaled-search`, **2nd and 3rd regular
  fights only** (floor ≥ 2, `category = 'easy'`), hash-selected and frozen in `fights.csv` (1,000 fights; ~930 after
  diverged replays).
- **Arms** (all at 8 particles; `search_salt` = seed):

| arm | seeds | purpose |
|---|---|---|
| MCTS 20k | 0–3 | reference; seed-vs-seed noise floor |
| MCTS 1k, 5k | 0–1 | saturation |
| oracle 20k | 0 | ceiling (converged at 10k on floor 1) |
| `act1-gen1` value net 20k | 0–1 | demonstrated fair headroom |

- **Analysis:** paired HP-eq with a 95% interval (bootstrap clustered by run seed), per encounter and pooled, with
  the four encounters weighted equally. Report the A/A SD and the fights needed for ±0.3 HP. Report deaths even if 0.
- **Impact:** ceiling impact = (oracle − MCTS) × 2.9 fights per run × carry-over 1 × X0.

## Decision (pre-registered)

- **Stop** if ceiling impact < 0.5 act-clear points. Expected: the ceiling is ~3–5 HP/fight, i.e. ~10–15 HP per
  run, so this hinges on X0. A plausible 0.2–0.5 points per HP puts it in grey or above. That would be the first
  real surprise.
- **If not stopped:** the demonstrated gain (net − MCTS) × 2.9 × X0 gives an achievable floor. If that is < 0.5
  points, deprioritise anyway: the rest of the ceiling is mostly the value of seeing the draws.
- **A shortcut worth knowing:** if MCTS 1k ≈ 20k, cheaper easy fights free compute for elites and bosses in the
  full-act harness. That alone is useful.

## Cost

Easy fights are ~2 turns; assume ~1 s/fight/worker at 20k (time a 20-fight pilot). 12 runs × ~1,000 fights on
11 workers ≈ **30–45 min**, plus X0.

## Risks and caveats

- **Carry-over of 1 overstates the value of HP:** rest sites heal, and HP above what's needed has no value. The
  stop rule is conservative in that direction.
- **The oracle can steer RNG** (it searches with the true RNG state), so the ceiling is loose. Fine for a stop
  decision.
