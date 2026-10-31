# NN100 simulation sweep — 2026-10-31

Checkpoint 132; same 500 paired starts, 50 per training deck. Eight particles, zero noise.

| Simulations | Wins / 500 | Win rate | Difference vs 256, pp [95% CI] | Process wall |
|---|---:|---:|---|---:|
| 16 | 264 | 52.8% | −6.8 [−10.8, −2.8] | 9.954 s |
| 32 | 285 | 57.0% | −2.6 [−6.3, +1.1] | 16.704 s |
| 64 | 292 | 58.4% | −1.2 [−5.0, +2.6] | 30.970 s |
| 128 | 312 | 62.4% | +2.8 [−0.7, +6.3] | 58.967 s |
| 256 (cached) | 298 | 59.6% | reference | 113.076 s |

Approximate paired-normal 95% intervals over 500 fights; exploratory, no multiplicity
correction. They do not measure unseen-deck generalization. Gains/losses versus 256:
16 sims 35/69; 32 sims 39/52; 64 sims 45/51; 128 sims 46/32.
One timing observation per budget, including native setup; variability unknown.
The 256 measurement is from the earlier leaf-mix run, not a same-session timing control.

| Deck | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|
| apparition-block | 40 | 42 | 43 | 47 | 47 |
| perfected-strike | 8 | 7 | 12 | 15 | 8 |
| clash | 12 | 15 | 14 | 13 | 11 |
| corruption-fnp | 14 | 12 | 13 | 12 | 11 |
| fire-breathing | 14 | 17 | 17 | 20 | 19 |
| limit-break-rage | 29 | 32 | 30 | 33 | 31 |
| barricade-searing | 50 | 50 | 50 | 50 | 50 |
| exhume-immolate | 44 | 47 | 47 | 48 | 49 |
| demon-combust | 30 | 34 | 35 | 40 | 36 |
| panache | 23 | 29 | 31 | 34 | 36 |

Artifacts: `/home/sborowsk/project/sts/worktrees/combat-leaf-mix-benchmark/runs/benchmark/sim-sweep-132-approved/`;
launcher log adjacent at `sim-sweep-132-approved.log`. Source revision: `d600bd1`.
All four processes succeeded: 500 matching unique IDs each, no capped fights.

**Recommendation:** use 128 simulations as the next provisional operating point: best
observed win count with approximately half the earlier 256-run wall time. This is not
a demonstrated strength improvement or non-inferiority guarantee. 64 is also a useful
lower-cost candidate; its interval still allows a material strength loss.
