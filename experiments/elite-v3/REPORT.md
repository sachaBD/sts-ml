# elite-v3 report

**Status: done** (2026-09-27 22:38Z → 2026-09-28 02:47Z, 4h09 of the 8h budget; every gate passed).
Plan: [PLAN.md](PLAN.md). Log: [LOG.md](LOG.md). Tables: [results/](results/).

## Headline

**The v3 net with policy priors at every search node is the first learned agent to match or edge out
guided-rollout MCTS on the Act 1 elite mix, but it doesn't clearly beat it.** On the 1,410 reserved final fights:
**+0.42 HP-eq/fight [95% CI −0.16, +0.98]**. By elite:
- Gremlin Nob +1.00 [+0.31, +1.70]: a real gain.
- Lagavulin +0.98 [−0.09, +2.03]: likely a gain.
- Three Sentries −0.72 [−1.85, +0.38]: still behind.

Wins were 1,282 vs 1,289 for MCTS (7 more deaths, almost all on Sentries).

The gain comes from **the policy-guided search, not from the value net alone**. The same net used as a value-only
leaf still loses: −1.46 [−2.78, −0.11] on validation, similar to the old v2 nets. That result mainly reflects
the value-only search, so it doesn't mean the new inputs don't matter.

## Results

**Validation** (421 of the 450 bucket-5 fights; 29 skipped because their replay diverged on the updated
simulator). Paired against MCTS re-run on the current simulator (20k simulations, 8 particles):

| Arm | Overall HP-eq [95% CI] | Nob | Lagavulin | Sentries | Wins (MCTS 376) |
|---|---:|---:|---:|---:|---:|
| t1, value-only leaf | −1.46 [−2.78, −0.11] | +0.08 | −2.24 | −2.24 | 365 |
| t1 + policy priors, c_puct 1.0 | +0.27 [−0.85, +1.42] | +1.03 | +0.32 | −0.53 | 373 |
| **t2 + policy priors, c_puct 1.0** | **+0.82 [−0.39, +2.07]** | +1.50 | −0.01 | +0.94 | 378 |
| t2 + policy priors, c_puct 0.5 | +0.82 [−0.37, +2.03] | +1.41 | +0.66 | +0.39 | 379 |
| t2 + policy priors, c_puct 2.0 | −0.01 [−1.20, +1.16] | +0.93 | −0.50 | −0.49 | 372 |
| t3 (1 round self-play) + policy, c 1.0 | +0.21 [−0.96, +1.39] | +1.42 | +0.57 | −1.35 | 370 |

t1 and t2 differ only in the policy-loss weight (0.05 vs 0.25). The stronger policy loss helped.

**Final test** (1,410 of the 1,500 reserved fights; 90 diverged): t2 + policy priors, c_puct 1.0, vs MCTS.

| Encounter | Fights | HP-eq [95% CI] | Wins MCTS / v3 | v3-only / MCTS-only wins |
|---|---:|---:|---:|---:|
| Gremlin Nob | 462 | +1.00 [+0.31, +1.70] | 441 / 442 | 4 / 3 |
| Lagavulin | 479 | +0.98 [−0.09, +2.03] | 445 / 445 | 9 / 9 |
| Three Sentries | 469 | −0.72 [−1.85, +0.38] | 403 / 395 | 10 / 18 |
| **All** | **1,410** | **+0.42 [−0.16, +0.98]** | 1,289 / 1,282 | 23 / 30 |

As expected, the final result is smaller than the validation figure (+0.82). The candidate was the best of
several arms on the same validation fights, which flatters its validation score.

## Data and training

- **Teacher data:** 4,884 fights (216 skipped because the replay diverged, 4.2%): 377k rows, 92.6k decisions.
  2,730 fights started with at least one potion, covering all 32 Ironclad potions (42–136 fights each); 64 distinct
  relics appeared. It ran 2.7× faster than planned (35 min).
- **Training:** about 5 minutes per net on the GPU. Validation value MSE was 0.0076 (t1) and 0.0072 (t2), against
  0.0069 for the teacher's own estimate and 0.039 for the baseline. The policy's top move matches the teacher's
  most-visited move 70–71% of the time.
- **Self-play:** 2,335 fights. The fine-tuned t3 did not improve play and was worse on Sentries.

## Issues found (for the next round)

1. **The potions-kept head overfits.** Its validation MSE is about 70× the training MSE, and it rises during
   agents.combat.value. Potion outcomes are sparse, so this head needs regularising or a smaller loss weight.
2. **Sentries is still the weak spot.** Priors turned that elite from a large loss into roughly even, but MCTS
   still wins more of those fights. This is where to focus next.
3. **Self-play run design:** t3's internal validation split overlapped t2's training data, so its training
   metrics look better than they are. One round with 2.3k fights is too little to judge expert iteration.
4. **Data loading is getting heavy:** the query that selects source fights took 24 minutes at ~14 GB RSS (swap
   full) before self-play started. The combat_v3 view needs pruning or partitioning before larger runs.
5. **The c_puct 0.5 and 1.0 arms tied exactly overall** (both add up to the same HP total over the 421 fights) but
   differ per elite. The tie rule picked 1.0; c = 0.5–1.0 looks like the right range and 2.0 is worse.
6. **One launch crashed on memory:** two app launchers ran their start-up queries at the same time and were
   killed for lack of memory. Relaunching the training alone fixed it. Stagger launches.
7. **Scaling note:** the search normalises values by 56 + max HP and the reports use 55 + max HP. Values are
   converted between the two, so this is consistent; it's only a note.

## Recommendation

Adopt the v3 net with policy priors (c_puct 0.5–1.0) as the elite agent candidate, but it doesn't yet meet the
experiment's success bar: overall is not significant, and Sentries is still negative. Next:
- More and more diverse elite training data; about half the compute budget is unused.
- Regularise or down-weight the potions-kept head.
- Look into Sentries specifically: the Dazed cards and Artifact handling.
- Rerun expert iteration with a clean split and more self-play fights.

## Deeper look: where the final-test difference comes from (added after the deck)

Paired final-test fights (1,410), split into subsets. **Difficulty proxy that doesn't depend on either compared
agent:** how the original data-generating teacher (5k-simulation MCTS, one random move) fared when it played the
identical start. Split points are exploratory and several were tried, so treat these as hypotheses.

| Subset | Fights | HP-eq v3 − MCTS [95% CI] | Deaths MCTS / v3 |
|---|---:|---:|---:|
| **Sentries, long** (original teacher > 6 turns) | 193 | **−2.31 [−4.34, −0.37]** | 17 / 26 |
| Sentries, short (≤ 6 turns) | 276 | +0.39 [−0.93, +1.70] | 49 / 48 |
| **Lagavulin, hard** (original lost ≥ 30 HP or died) | 264 | **+2.35 [+0.78, +3.97]** | 34 / 31 |
| Lagavulin, easy (original lost < 10 HP) | 42 | −4.12 [−7.26, −1.19] | 0 / 1 |
| Nob, each difficulty band | 65–202 | +0.57 to +1.24 | 21 / 20 total |
| All, short fights (≤ 4 turns, MCTS length) | 775 | +0.81 [+0.24, +1.39] | 60 / 57 |
| All, long fights | 635 | −0.06 [−1.12, +0.94] | 61 / 71 |

- **The Sentries deficit is entirely in long fights.** It holds whichever way "long" is defined: by the original
  teacher's length (−2.31) or by the longer of the two arms (−1.70). In short Sentries fights v3 is even or ahead.
- **Opening targeting is not the difference.** In turns 0–1 both agents put 86–89% of single-target attacks on one
  sentry, almost always a side one (MCTS right/left/middle 216/185/59; v3 220/192/50). They pick the same main
  target in 65% of fights. The few fights where v3 focuses the middle sentry look bad (−2.80, n = 50), but the
  sample is too small to conclude anything.
- So v3 falls behind in the **middle and late game of long Sentries fights**, where Dazed cards pile up and the
  horizon is long. Guided rollouts play to the end of the fight; the value net has to predict it from far away.
- **Lagavulin:** v3 gains a lot where the fight is hard but loses where it is easy (n = 42, small). Possibly it
  doesn't rush the kill before Lagavulin wakes. Worth replaying a few of those fights to check.
