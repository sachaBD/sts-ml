# elite-bench analysis (bsmoke)

Runs found: mcts-1k (salts [0]; 12 fights), mcts-20k (salts [0, 1, 4]; 12 fights), mcts-20k-p32 (salts [0]; 4 fights), mcts-5k (salts [0]; 12 fights), oracle-20k (salts [0]; 12 fights), oracle-50k (salts [0]; 12 fights), v3-20k (salts [0, 1, 4]; 12 fights), v3-5k (salts [0]; 12 fights)

## A. Headroom: oracle (sees draws and RNG) vs fair play, 20k simulations

Fair columns: mean over salts 0-3 (per-fight expected value). Unwinnable = the oracle lost. Pivotal = some fair 20k run lost and the fight is winnable (oracle or some fair run won). HP-eq lost = 35 + starting HP - HP-eq (the most a perfect player could save).

| elite | fights | win oracle | win MCTS | win v3 | HP-eq lost oracle | HP-eq lost MCTS | HP-eq lost v3 | unwinnable | pivotal | unwinnable but MCTS won (sanity) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 12 | 100.0% | 91.7% | 79.2% | 28.3 | 38.0 | 41.1 | 0 (0.0%) | 4 (33.3%) | 0 |
| gremlin_nob | 4 | 100.0% | 87.5% | 75.0% | 36.7 | 45.2 | 49.6 | 0 (0.0%) | 1 (25.0%) | 0 |
| lagavulin | 4 | 100.0% | 87.5% | 75.0% | 30.7 | 43.5 | 45.5 | 0 (0.0%) | 2 (50.0%) | 0 |
| three_sentries | 4 | 100.0% | 100.0% | 87.5% | 17.5 | 25.2 | 28.2 | 0 (0.0%) | 1 (25.0%) | 0 |

Oracle − MCTS HP-eq per fight (upper bound on fair headroom; includes the value of seeing the draws): +9.67 [+5.67, +14.33]

Oracle convergence (subset, n=12): 50k − 20k = +1.58 [+0.00, +3.67] HP-eq, wins 12 → 12

## B. Noise floor: the same agent under different search salts (A/A)

Each row: two runs of the listed arms on identical fights. 'discordant wins' = fights won by only one run (first / second). An A/A row shows how much of a single-run comparison is pure chance.

| pair | fights | mean diff | SD of diff | identical score | discordant wins |
|---|---:|---:|---:|---:|---:|
| mcts-20k s0 vs mcts-20k s1 | 12 | -5.00 | 14.5 | 25% | 2 / 0 |
| v3-20k s0 vs v3-20k s1 | 12 | -3.25 | 12.2 | 58% | 1 / 0 |
| mcts-20k s0 vs v3-20k s0 | 12 | -4.00 | 15.1 | 25% | 2 / 0 |
| mcts-20k s1 vs v3-20k s1 | 12 | -2.25 | 19.8 | 33% | 2 / 1 |

MCTS 20k, 2 salts, 12 fights: within-fight (salt) variance 117 of total 474 HP-eq² (25%); outcome (win/loss) varies across salts in 2 fights (16.7%). Fights never won: 0, always won: 10.

## C. v3 (policy priors) vs MCTS, 20k simulations: per-fight means over salts

Δ = v3 − MCTS HP-eq per fight. 'K=1' uses salt 0 only (the old protocol); 'all salts' uses salts 0-3 on every fight. Winnable = the oracle won.

| elite | fights | Δ HP-eq, K=1 | Δ HP-eq, all salts | Δ HP-eq, winnable only | win prob MCTS / v3 |
|---|---:|---:|---:|---:|---:|
| all | 12 | -4.00 [-13.67, +3.33] | -3.13 [-11.88, +4.54] | -3.13 [-11.79, +4.46] | 91.7% / 79.2% |
| gremlin_nob | 4 | -8.75 [-27.00, +0.75] | -4.38 [-13.50, +0.37] | -4.38 [-13.50, +0.37] | 87.5% / 75.0% |
| lagavulin | 4 | -7.25 [-27.75, +4.50] | -2.00 [-26.25, +15.25] | -2.00 [-26.25, +15.25] | 87.5% / 75.0% |
| three_sentries | 4 | +4.00 [+0.25, +8.25] | -3.00 [-16.00, +4.25] | -3.00 [-16.00, +4.25] | 100.0% / 87.5% |

- three_sentries long (>6 turns, MCTS s0): n=3, Δ +3.50 [+2.00, +5.00]
- three_sentries short: n=1, Δ -22.50 [+nan, +nan]
- lagavulin long (>6 turns, MCTS s0): n=2, Δ -9.00 [-37.00, +19.00]
- lagavulin short: n=2, Δ +5.00 [+4.00, +6.00]

## C2. Pivotal fights on held-out salts 4-7 (chosen from salts 0-3; no selection bias)

| elite | fights | Δ v3 − MCTS (salts 4-7) | MCTS win prob salts 0-3 → 4-7 | v3 win prob 0-3 → 4-7 | SD of per-fight Δ |
|---|---:|---:|---:|---:|---:|
| all | 4 | -12.25 [-45.00, +20.50] | 75.0% → 50.0% | 37.5% → 25.0% | 35.8 |
| gremlin_nob | 1 | +0.00 [+nan, +nan] | 50.0% → 0.0% | 0.0% → 0.0% | 0.0 |
| lagavulin | 2 | -2.50 [-46.00, +41.00] | 75.0% → 50.0% | 50.0% → 50.0% | 43.5 |
| three_sentries | 1 | -44.00 [+nan, +nan] | 100.0% → 100.0% | 50.0% → 0.0% | 0.0 |

The drop from salts 0-3 to 4-7 is regression to the mean: how much of 'pivotal' was luck in the selection runs.

## D. Budget curves (paired, vs MCTS 20k on the same fights)

Mean over available salts per fight (MCTS 20k: all its salts). 5k/50k-v3 and oracle-50k arms ran on the 600-fight subset only; compare rows with the same fight count.

| arm | elite | fights | salts | Δ HP-eq vs MCTS 20k | win prob MCTS 20k → arm |
|---|---:|---:|---:|---:|---:|
| mcts-1k | all | 12 | [0] | -8.75 [-20.17, +1.00] | 91.7% → 75.0% |
| mcts-1k | gremlin_nob | 4 | [0] | -16.25 [-35.00, -0.25] | 87.5% → 50.0% |
| mcts-1k | lagavulin | 4 | [0] | +2.50 [-9.00, +14.00] | 87.5% → 100.0% |
| mcts-1k | three_sentries | 4 | [0] | -12.50 [-33.38, +0.25] | 100.0% → 75.0% |
| mcts-5k | all | 12 | [0] | -3.92 [-8.79, +0.58] | 91.7% → 83.3% |
| mcts-5k | gremlin_nob | 4 | [0] | -4.50 [-13.50, +0.25] | 87.5% → 75.0% |
| mcts-5k | lagavulin | 4 | [0] | -5.50 [-16.50, +5.50] | 87.5% → 75.0% |
| mcts-5k | three_sentries | 4 | [0] | -1.75 [-4.13, +0.87] | 100.0% → 100.0% |
| v3-5k | all | 12 | [0] | -6.58 [-16.79, +2.42] | 91.7% → 75.0% |
| v3-5k | gremlin_nob | 4 | [0] | -4.25 [-13.38, +0.50] | 87.5% → 75.0% |
| v3-5k | lagavulin | 4 | [0] | -2.75 [-26.25, +14.50] | 87.5% → 75.0% |
| v3-5k | three_sentries | 4 | [0] | -12.75 [-34.13, +0.37] | 100.0% → 75.0% |

v3 − MCTS at equal budget (subset fights where both budgets exist):

| budget | fights | Δ v3 − MCTS | mean HP-eq MCTS / v3 |
|---|---:|---:|---:|
| 5k | 12 | -2.67 [-13.67, +8.42] | 47.9 / 45.3 |
| 20k | 12 | -2.86 [-12.89, +5.89] | 51.1 / 48.2 |

## E. Particles: MCTS 20k with 32 (salts 0-3) vs 8 particles (held-out salts 4-7), pivotal fights

| elite | fights | Δ HP-eq 32 − 8 particles | win prob 8 → 32 |
|---|---:|---:|---:|
| all | 4 | -12.75 [-34.50, +0.00] | 50.0% → 25.0% |
| gremlin_nob | 1 | +0.00 [+nan, +nan] | 0.0% → 0.0% |
| lagavulin | 2 | -23.00 [-46.00, +0.00] | 50.0% → 0.0% |
| three_sentries | 1 | -5.00 [+nan, +nan] | 100.0% → 100.0% |

## F. Power: paired fights needed to detect a true difference (80% power, two-sided 5%)

| protocol | SD of per-fight Δ | n for 0.5 HP-eq | n for 1.0 | n for 2.0 |
|---|---:|---:|---:|---:|
| K=1 salt per fight | 15.1 | 7109 | 1778 | 445 |
| all salts per fight | 14.5 | 6629 | 1658 | 415 |

