# Combat headroom: is more combat research worth it? (easy, hard, boss)

> **Status: DESIGN ONLY, NOT ACTIONED.** Nothing here has run. Written 2026-09-29 after elite-bench.
> When a study runs, give it a directory `experiments/act-1-combat/<date>-<name>/` and link it here.

**Question:** for each act-1 fight category, is MCTS (guided rollouts, 20k simulations, 8 particles) close enough
to optimal that further combat research is worth less than macro work (pathing, card choice, rests)?

| Study | Hypothesis | Plan |
|---|---|---|
| Easy: a run's first 3 regular fights (Cultist, Jaw Worm, Two Louse, Small Slimes) | MCTS is effectively optimal; stop | [easy/PLAN.md](easy/PLAN.md) |
| Hard: every later regular fight (10 encounters) | Probably the same as easy; unsure | [hard/PLAN.md](hard/PLAN.md) |
| Boss (Slime Boss, Guardian, Hexaghost) | Behaves like the elites | [boss/PLAN.md](boss/PLAN.md) |
| X0: HP exchange rate (shared) | Converts HP saved into act-clear points | below |

## Why per-run impact, not per-fight

A gain only matters as much as it changes the run. In the act-1 evaluation (`act1-eval-mcts-a20`, 1,000 runs,
MCTS at 500 / 2k / 5k / 15k simulations for easy / hard / elite / boss):

| category | fights per run | death rate | HP lost per won fight | deaths per 1,000 runs |
|---|---:|---:|---:|---:|
| easy | 2.9 | 0.1% | 4–8 | ~2 |
| hard | 1.6 | 0.6% | 4–15 | ~9 |
| elite | 1.3 | 6.5% | 22–27 | ~83 |
| boss | 0.89 (reached) | 32% (Hexaghost 43%) | 19–37 | ~285 |

The boss is where runs end. Easy and hard fights matter only through the **HP they cost**. How much that HP is
worth is the exchange rate X0.

## Decision rule (same for every study)

Every study measures three things, all paired on identical fights (see
[../evaluation_methodology.md](../evaluation_methodology.md)):

1. **Ceiling:** the oracle MCTS, which sees draws and RNG, minus MCTS. It is an upper bound on any fair
   improvement, because it includes the value of information no fair player has.
2. **Saturation:** the MCTS budget curve. If more simulations don't help, better search settings alone won't
   either.
3. **Demonstrated fair headroom:** the best available fair challenger (a net) minus MCTS. A lower bound on what
   is achievable.

**Ceiling impact** = ceiling (HP-eq per fight) × fights per run × carry-over × X0 (act-clear points per HP).
Carry-over is the share of HP saved that is still there at the boss; use 1 as the upper bound.

- **Stop** (combat research not worth it for this category) if the ceiling impact is < 0.5 act-clear points per
  run. The fair optimum can only be lower, so the conclusion holds whatever it is.
- **Continue** if the *demonstrated* fair headroom is ≥ 1 act-clear point per run.
- Otherwise the category is **grey**: rank it against macro by cost.

## X0: the HP exchange rate (shared prerequisite)

**Goal:** how many act-clear points one extra HP at the boss is worth (dP(win)/dHP), per boss.

- **Method:** replay boss fights with starting HP randomised around the stored value (`fight_resample`,
  `hp_sd = 12`, 3 samples per fight) with MCTS at 20k. Then fit a logistic regression of win on starting HP, with
  a per-fight intercept (or pair the samples of one fight). HP is randomised, so the slope is causal and not
  confounded by deck.
- **Fights:** 200 per boss from buckets 0–1 of `act1-all-bosses-a20-scaled-search`, the same as the boss study,
  so it can reuse those fight lists.
- **Report:** slope per boss with a bootstrap CI (clustered by run seed), and act-clear points per HP =
  slope × P(reach boss) (≈ 0.89).
- **Carry-over:** 1 as the upper bound. If a study lands in grey because of it, measure it causally: the full-act
  harness with a fixed HP penalty after floor 3 vs none, 1,000 paired seeds. That is not designed here.
- **Compute:** about 1,800 boss fights at ~10–15 s/fight/worker on 11 workers ≈ 30–40 min. Time a 20-fight pilot
  first.

## Shared prerequisites (small code work, before any study)

- **Generalise the elite-bench scripts** (`experiments/elite-bench/`), which hard-code the three elites:
  - `bench.py`: `ELITES` → encounters read from the data.
  - `make_fights.py`: a category and per-encounter count, with an optional floor filter.
  - `make_configs.py`: category, fight list and arms from arguments.
  - `analyze.py`: sections by encounter; a "per-run impact" line using the table above and X0.
- **Fight splits:**
  - buckets 0–1 of `act1-all-bosses-a20-scaled-search`: dev benchmark. No net trained on them. Elites there are
    already used; non-elite fights are fresh apart from the old `ab-*-dev` evaluations.
  - buckets 2–3: confirmation. Non-elite fights there are untouched.
  - About 5–7% of stored fights no longer replay identically on the current simulator; `skip_diverged = true`.
- **Challenger nets:** the only policy-prior (v3) nets are elite-only. Non-elite challengers today are value-only:
  `value_net_v1/2026-09-26/act1-gen1` (+0.57 HP/fight over MCTS on floor-1 easy fights) and
  `value_net_v1/2026-09-27/ab-gen1` (+5.5 on Slime Boss, earlier protocol). A v3 policy net for a new category means
  a teacher-data run and a training (the elite-v3 / elite-bench t5 recipe).

## Prior evidence (read first)

- `experiments/act-1-easy-combats/README.md`: floor-1 easy fights only.
  - MCTS stops improving at about 1k simulations; 64 particles don't help.
  - The gen1 net beats MCTS by +0.57 HP/fight; the oracle ceiling is +3.3 HP/fight. No deaths.
- `experiments/elite-bench/results/`: elites.
  - The oracle wins 96.7% against MCTS's 91%; MCTS saturates around 10k simulations; 32 particles don't help.
  - The t5 net beats MCTS by +0.83 HP-eq on fresh fights. Being close to MCTS is **not** evidence of being close
    to optimal.
- `experiments/act-1-combat/2026-09-26-all-bosses/REPORT.md` and `2026-09-27-act1-eval/README.md`: bosses under
  the old single-seed protocol. Slime Boss +5.5 for gen1; Hexaghost ~54% for both agents.
