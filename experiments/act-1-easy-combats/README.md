# Act 1, floor 1 easy-pool fights: is MCTS near-optimal?

**Question:** with enough simulations, does our MCTS (the guided-rollout teacher) play the floor-1 easy fights
(Cultist, Jaw Worm, Two Louse, Small Slimes) close to optimally?

**Answer: no.** It is close, but measurably below optimal, and more compute doesn't close the gap.

- **Budget doesn't help.** Fair MCTS is flat from 1k to 100k simulations and from 8 to 64 belief particles
  (all within ±0.2 HP per fight, pooled). It stops improving at about 1k simulations.
- **The plateau is not optimal.** A different fair player, the gen1 value-net search at 20k simulations, beats
  100k-simulation MCTS by **+0.57 HP per fight** (95% CI +0.25 to +0.88, p ≈ 0.0004, n = 400 paired fights). Most
  of that comes from Jaw Worm (+1.3) and Small Slimes (+0.8).
- **Upper bound on what's left:** a clairvoyant oracle loses **3.3 HP per fight less** than fair MCTS (CI 3.0 to
  3.6). Part of that is the value of knowing the draws, which no fair player can get. So the fair optimum lies
  somewhere between +0.6 and +3.3 HP per fight above MCTS. We only know the lower end.
- **Likely cause:** the leaf evaluation, not search size. On Jaw Worm, 1k simulations beat 100k by +1.0 HP (CI
  +0.2 to +1.9, p = 0.014; the pilot pointed the same way). That is what search converging on a biased rollout
  value looks like. It's a hint, not a proven cause.
- **In practice:** nobody dies (0 deaths in 3,120 fight plays), and fair MCTS loses 5–12 HP per fight. At about 3
  easy fights per run, the demonstrated headroom is about 2 HP per act. That matters little next to elites and the
  boss.

## Results (main study: 400 fights, 100 per encounter)

Mean HP lost per fight (all won):

| player | cultist | jaw worm | two louse | small slimes |
|---|---:|---:|---:|---:|
| MCTS 1k sims | 6.3 | 11.5 | 5.6 | 8.6 |
| MCTS 10k | 6.0 | 12.5 | 5.4 | 8.4 |
| MCTS 100k | 5.9 | 12.5 | 5.3 | 8.9 |
| MCTS 100k, 64 particles | 6.0 | 12.3 | 5.3 | 8.4 |
| gen1 value net, 20k | 5.8 | 11.2 | 5.3 | 8.1 |
| oracle MCTS 100k (clairvoyant) | 2.4 | 6.8 | 4.5 | 5.8 |

Paired difference vs MCTS 100k, HP per fight (positive means better). Pooled means the 4 encounters weighted
equally. Brackets are 95% CIs:

| player | cultist | jaw worm | two louse | small slimes | pooled |
|---|---|---|---|---|---|
| MCTS 1k | −0.42 [−0.95, +0.11] | **+1.03** [+0.21, +1.85] | −0.25 [−0.52, +0.02] | +0.35 [−0.16, +0.86] | +0.18 [−0.11, +0.46] |
| MCTS 10k | −0.12 [−0.44, +0.20] | +0.01 [−0.74, +0.76] | −0.10 [−0.29, +0.09] | +0.51 [+0.07, +0.95] | +0.07 [−0.16, +0.31] |
| MCTS 100k, 64 particles | −0.10 [−0.48, +0.28] | +0.23 [−0.47, +0.93] | +0.05 [−0.11, +0.21] | +0.53 [−0.02, +1.08] | +0.18 [−0.07, +0.42] |
| **gen1 value net 20k** | +0.13 [−0.25, +0.51] | **+1.32** [+0.26, +2.38] | −0.03 [−0.19, +0.13] | **+0.84** [+0.28, +1.40] | **+0.57** [+0.25, +0.88] |
| oracle 100k | +3.46 [+2.91, +4.01] | +5.78 [+4.95, +6.61] | +0.76 [+0.46, +1.06] | +3.17 [+2.51, +3.83] | +3.29 [+2.98, +3.60] |

Full tables, including win/loss counts per fight: `results/`.

## Method

- **Fights:** floor-1 fights from the stored A20 Ironclad runs (`combat_v3/2026-09-24/act1-a20-8`), dev split only
  (`run_seed % 10 in (0, 1)`), so the gen1 net never trained on them. The deck is the starter deck plus whatever
  Neow gave, at 68/80 HP. Fights are sampled at random within each encounter (ordered by `md5(episode_id)`).
- **Design:** every player replays the same starting states with the same game RNG seed, and results are paired
  fight by fight. Search is deterministic given the state, so there is one result per fight and player.
- **Score:** HP-eq. That is final HP, plus 4 per potion, or −35 on a death. At floor 1 this is just HP (no potions,
  no deaths).
- **Sample size:** a pilot (30 per encounter, ranks 3–32, not reused) measured a paired SD of about 2.6 HP between
  fair players. 100 per encounter then gives a pooled 95% CI of about ±0.3 HP, which is enough to tell "optimal"
  from a 1-HP gap. The pilot also tested 32 particles and oracle 10k, and its conclusions match the main study
  (`results/pilot-*.md`).
- **CIs:** normal approximation on paired differences, with encounters as independent strata. The p-values are
  not corrected for multiple comparisons. The headline (gen1 vs MCTS pooled, p ≈ 0.0004) survives Bonferroni over
  the roughly 25 cells tested. The per-encounter Jaw Worm results (p ≈ 0.014) don't, and are supporting evidence
  only.
- **Compute:** about 30 min of wall time on 12 cores in total, pilot included.

## Caveats

- **The oracle is only an upper bound.** It sees the draw order and the RNG, and with max backup it can also steer
  RNG consumption. Its gap to fair play mixes in the value of information, so it doesn't measure MCTS error. It
  has converged (10k vs 100k: −0.15 HP in the pilot).
- **gen1 is only a lower bound on the headroom.** It has a different leaf and a different budget, and it shows
  that better fair play exists. It doesn't show how much.
- **Only floor 1.** Starter deck, full HP. Easy fights later in the act (hand-built decks, lower HP) are untested.
- The "biased rollout leaf" explanation is inferred from the budget curves. It hasn't been tested directly.

## Files

```
make_configs.py   writes configs/*.toml (fight sample + player settings; the run list)
configs/          one value_play config per run (command on line 1)
run_all.sh        runs configs in sequence: ./experiments/act-1-easy-combats/run_all.sh NAME...
analyze.py        paired tables: analyze.py PREFIX [--ref RUN_ID]
results/          analyze.py output (main + pilot, vs oracle and vs MCTS 100k)
logs/             run logs
```

Runs: `combat_v3/2026-09-28/easy-pilot-*` and `combat_v3/2026-09-28/easy-main-*`.

Reproduce (from the repo root):

```bash
./experiments/act-1-easy-combats/run_all.sh main-fair1k main-fair10k main-fair100k main-fair100k-p64 main-gen1net20k main-oracle100k
PYTHONPATH=python .venv/bin/python experiments/act-1-easy-combats/analyze.py easy-main-
PYTHONPATH=python .venv/bin/python experiments/act-1-easy-combats/analyze.py easy-main- --ref easy-main-fair100k
```
