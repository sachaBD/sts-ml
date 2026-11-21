# Search scaling: depth (sims) vs breadth (worlds)

Tracking issue: #202. Prior result: root Q is particle-noise-limited at 256 × 8 (#198).

**Question.** For a given network budget, which (sims, worlds) gives the best move and the best training target?

Scripts here; run each as `.venv/bin/python apps/search_scaling/<script>.py` (usage in its docstring).

**Fixed for every experiment.**
- Network: wide128 `…-256s-graph-cyclefix/out/training/iter030` (validation 1608/2000).
- Search: gumbel_scale 0, value_scale 0.1, no rollouts.
- Code: the commit that adds each experiment's results below.

**Metric.**
- Regret per decision = Q_ref(reference best) − Q_ref(chosen), in win points, against the reference set below.
- Also reported: agreement with the reference move, TV(target, reference target), and network requests per decision.
- A cell never shares worlds with the reference: the reference uses draws 1 000 001–2, cells use draws < 1 000 000.

Data lives in `runs/schema=search_scaling_v1/date=2026-11-21/`, not in git.

## R0: reference regret set (`reference.py`)

| | |
|---|---|
| Positions | 2000 fights sampled from iter030's starts (seed 0), each walked by the agent (256 × 8, gumbel 0, play's draw). Each fight contributes **one** decision with 2+ legal actions, chosen uniformly, so positions are independent |
| Reference | 2048 sims × 64 worlds, all root actions considered (`considered_actions` 128), on 2 independent draws |
| Cost | 119 ms per reference search on GPU; ~25 min in total |
| Output | `id=reference-2048x64/`: `positions.jsonl`, `reference.jsonl` (root edges per draw), `graphs.jsonl.gz` (every full search graph, ~1 GB), `meta.json` |

Built in 46 min (walk 31, references 15): 2000 positions, 1.3 GB of graphs. `r0_stability.py`:

| | |
|---|---|
| Reference Q SD per action (one draw) | median 0.0013, p90 0.021 |
| Move agreement, draw A vs B | 93.0% |
| Regret of one draw judged by the other | **0.0028** ± 0.0003 |
| Regret of the network alone (0 sims) | **0.0084** ± 0.0010 (zero in 62% of positions) |

Stakes (gap between the best and second move) are concentrated:

| Gap | Share of positions | Share of the network's regret |
|---|---|---|
| < 0.01 | 79.5% | 39% |
| 0.01–0.05 | 13.8% | 27% |
| ≥ 0.05 | 6.7% | 33% |

**Decision: the reference is usable.** Its noise is independent of any cell's choice, so it adds variance to regret but no bias to *differences* between cells. Cells are compared by paired differences; their absolute regret includes a floor near the reference's own noise.

## E0: pilot (`grid.py`, `regret.py`)

200 positions; grid {64, 256, 1024} sims × {1, 8, 64} worlds; draw 1 for every cell.

Rule (set in the code before the run): n for E1 = (SD of paired difference vs 256 × 8 / 0.00025)².

| sims \ worlds | 1 | 8 | 64 |
|---|---|---|---|
| 0 (network) | 0.0064 | | |
| 64 | 0.0055 | 0.0056 | 0.0055 |
| 256 | 0.0052 | **0.0046** | 0.0053 |
| 1024 | 0.0051 | 0.0033 | **0.0018** |

Regret is in win points; SE ≈ 0.0019 at 64–256 sims and 0.0005–0.0012 at 1024. Network requests per search ≈ 0.85–0.9 × sims.

- **SD of paired differences:** 0.008–0.011 for cells at ≤ 256 sims, giving n ≈ 1100–2000. For 1024-sim cells it is 0.023–0.027, giving n ≈ 8500–11300. The tails dominate: a few high-stakes positions carry most of the variance.
- **Hint, not yet a result:** worlds only matter once sims are high. At 1024 sims, 1 → 8 → 64 worlds gives 0.0051 → 0.0033 → 0.0018.
- **Caveat:** 1024 × 64 is the cell closest to the reference's own configuration. Part of its low regret may be shared network bias, which E2 checks.

**Decision: run E1 on all 2000 positions.** Paired SEs come out ≈ 0.0002 for ≤ 256-sim cells and ≈ 0.0005 for 1024-sim cells, which resolves effects ≥ 0.0015 at high sims. If E1 needs finer resolution there, extend the reference set (46 min per 2000 positions).

## E1: grid (`grid.py`, `regret.py`, `e1_summary.py`)

2000 positions; sims {16 … 1024} × worlds {1 … 64}, one draw (1); 25 min. Regret in win points per decision (SE ≈ 0.0003–0.0007).

| sims \ worlds | 1 | 2 | 4 | 8 | 16 | 32 | 64 | requests |
|---|---|---|---|---|---|---|---|---|
| 0 (network) | 0.0084 | | | | | | | 1 |
| 16 | 0.0064 | 0.0065 | 0.0062 | 0.0061 | 0.0066 | 0.0063 | 0.0065 | 13 |
| 64 | 0.0059 | 0.0051 | 0.0053 | 0.0053 | 0.0050 | 0.0053 | 0.0052 | 55 |
| 256 | 0.0057 | 0.0047 | 0.0048 | **0.0047** | 0.0040 | 0.0041 | 0.0041 | 225 |
| 512 | 0.0064 | 0.0055 | 0.0049 | 0.0035 | 0.0034 | 0.0033 | 0.0032 | 445 |
| 1024 | 0.0064 | 0.0053 | 0.0050 | 0.0031 | 0.0029 | 0.0028 | 0.0027 | 870 |

Paired against 256 × 8 (negative = better):

| Cell | Δ regret | z |
|---|---|---|
| 0 sims | +0.0037 | +4.4 |
| 256 × 1 | +0.0010 | +1.8 |
| 256 × 16 | −0.0007 | −1.6 |
| 512 × 8 | −0.0012 | −3.2 |
| 512 × 16 | −0.0013 | −3.1 |
| 1024 × 8 | −0.0016 | −3.4 |
| 1024 × 16 | −0.0018 | −4.0 |
| 1024 × 64 | −0.0020 | −4.5 |

Regret by stakes (gap between the reference's best two moves):

| Cell | gap < 0.01 (80%) | 0.01–0.05 (14%) | ≥ 0.05 (7%) |
|---|---|---|---|
| 0 sims | 0.0042 | 0.0165 | 0.0421 |
| 256 × 8 | 0.0018 | 0.0112 | 0.0261 |
| 1024 × 1 | 0.0024 | 0.0156 | 0.0359 |
| 1024 × 16 | 0.0014 | 0.0084 | **0.0093** |

**Findings**
1. **Depth pays only with breadth.** With ≥ 8 worlds, regret falls steadily with sims. With 1–4 worlds it stops falling past 256 sims, and with 1 world it rises again: deep search overfits the futures it sampled.
2. **Breadth is nearly free.** Requests grow ~1% from 8 to 64 worlds. At 256 sims, 8 → 16 worlds is −0.0007 (z −1.6, not significant).
3. **The gain is in high-stakes decisions.** 1024 × 16 cuts regret on the 7% of positions with gap ≥ 0.05 from 0.026 to 0.009; on near-ties it barely moves.
4. **Scale:** a fight has ~50 decisions with 2+ actions. Summing regret predicts 64 × 8 → 256 × 8 is worth +2.7 win points per fight. Training's validation for this same checkpoint measured **+2.85** (iter030: 1551 → 1608 of 2000), so summed regret is roughly calibrated here (an older checkpoint measured +1.5). On that calibration, 1024 × 16 vs 256 × 8 predicts up to **+9 win points**, and 512 × 16 +6.
5. **Caveat:** cells nearer the reference's configuration (2048 × 64) share its network and its search, so part of their advantage may be shared bias. E2 decides.

## Plan
- **E2 confirm**: paired win-rate games on validation (2000 fights, same seeds per arm) for 256 × 8 (baseline), 256 × 16, 512 × 16 and 1024 × 16. Success: a lower bound > 0 on the paired difference vs baseline. Cost from training's timing (633 s per 2000 fights at 256 sims): ~85 min for the four arms.
