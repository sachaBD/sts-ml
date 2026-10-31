# Leaf-mix results — 2026-10-31

500 paired starts, 50 per training deck; checkpoint 132. One run per new variant.

| NN / rollout | Wins / 500 | Win rate | Difference vs NN50, percentage points [95% CI] | Process wall |
|---|---:|---:|---|---:|
| 50 / 50 (cached) | 288 | 57.6% | reference | historical; not comparable |
| 75 / 25 | 294 | 58.8% | +1.2 [−2.9, +5.3] | 116.046 s |
| 100 / 0 | 298 | 59.6% | +2.0 [−1.8, +5.8] | 113.076 s |

Intervals are approximate paired-normal 95% intervals over 500 fights, not unseen-deck
uncertainty. Gains/losses vs NN50: NN75 57/51; NN100 51/41. Neither difference excludes zero.
Timing includes native-process setup; n=1 per variant, timing variability unknown.

| Deck | NN50 | NN75 | NN100 |
|---|---:|---:|---:|
| apparition-block | 46 | 45 | 47 |
| perfected-strike | 11 | 11 | 8 |
| clash | 13 | 14 | 11 |
| corruption-fnp | 11 | 11 | 11 |
| fire-breathing | 17 | 17 | 19 |
| limit-break-rage | 28 | 33 | 31 |
| barricade-searing | 50 | 50 | 50 |
| exhume-immolate | 49 | 49 | 49 |
| demon-combust | 30 | 33 | 36 |
| panache | 33 | 31 | 36 |

Artifacts: `/home/sborowsk/project/sts/worktrees/combat-leaf-mix-benchmark/runs/benchmark/leaf-mix-132-approved/`.
Launcher log: adjacent `leaf-mix-132-approved.log`. Source revision: `d7ae7d0`.
Both processes succeeded with all 500 matching unique fight IDs and no capped fights.
GPU: RTX 3060; idle at launch. Input hashes and exact commands are in the artifacts.

**Conclusion:** NN100 has the highest observed win count and lowest measured wall time,
but this sample does not establish a strength advantage. Historical simulator and
training-set caveats remain as described in the README.
