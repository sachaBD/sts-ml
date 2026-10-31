# NN100 simulation sweep — 2026-10-31

Checkpoint 132; same 500 paired starts, 50 per training deck. Eight particles, zero noise.

| Simulations | Wins / 500 | Win rate [95% CI] | Difference vs 256, pp [95% CI] | Process wall |
|---|---:|---|---|---:|
| 16 | 264 | 52.8% [48.4, 57.1] | −6.8 [−10.8, −2.8] | 9.954 s |
| 32 | 285 | 57.0% [52.6, 61.3] | −2.6 [−6.3, +1.1] | 16.704 s |
| 64 | 292 | 58.4% [54.0, 62.6] | −1.2 [−5.0, +2.6] | 30.970 s |
| 128 | 312 | 62.4% [58.1, 66.5] | +2.8 [−0.7, +6.3] | 58.967 s |
| 256 (cached) | 298 | 59.6% [55.2, 63.8] | reference | 113.076 s |
| 512 | 304 | 60.8% [56.5, 65.0] | +1.2 [−2.5, +4.9] | 212.525 s |
| 1024 | 311 | 62.2% [57.9, 66.3] | +2.6 [−1.2, +6.4] | 422.696 s |

Wilson 95% win-rate intervals and approximate paired-normal 95% difference intervals over
500 fights; exploratory, no multiplicity correction. They describe evaluated-start
variability, not unseen-deck generalization. Gains/losses versus 256:
16 sims 35/69; 32 sims 39/52; 64 sims 45/51; 128 sims 46/32; 512 sims 48/42; 1024 sims 54/41.
One timing observation per budget, including native setup; variability unknown.
The 256 measurement is from the earlier leaf-mix run, not a same-session timing control.

| Deck | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|
| apparition-block | 40 | 42 | 43 | 47 | 47 | 50 | 50 |
| perfected-strike | 8 | 7 | 12 | 15 | 8 | 10 | 14 |
| clash | 12 | 15 | 14 | 13 | 11 | 13 | 17 |
| corruption-fnp | 14 | 12 | 13 | 12 | 11 | 16 | 12 |
| fire-breathing | 14 | 17 | 17 | 20 | 19 | 21 | 19 |
| limit-break-rage | 29 | 32 | 30 | 33 | 31 | 34 | 35 |
| barricade-searing | 50 | 50 | 50 | 50 | 50 | 50 | 50 |
| exhume-immolate | 44 | 47 | 47 | 48 | 49 | 44 | 48 |
| demon-combust | 30 | 34 | 35 | 40 | 36 | 31 | 33 |
| panache | 23 | 29 | 31 | 34 | 36 | 35 | 33 |

Artifacts: `/home/sborowsk/project/sts/worktrees/combat-leaf-mix-benchmark/runs/benchmark/sim-sweep-132-approved/`;
launcher log adjacent at `sim-sweep-132-approved.log`. Source revision: `d600bd1`.
The approved extension is in adjacent `sim-sweep-high-132-approved/` (and `.log`),
source revision `1e82d5a`. All six processes succeeded: 500 matching unique IDs each,
no capped fights.

**Recommendation:** use 128 simulations as the next provisional operating point: best
observed win count with approximately half the earlier 256-run wall time. This is not
a demonstrated strength improvement or non-inferiority guarantee. The 512/1024 extension
shows no observed advantage over 128 despite substantially greater wall time; uncertainty
still permits differences. 64 is also a useful lower-cost candidate; its interval still
allows a material strength loss.
