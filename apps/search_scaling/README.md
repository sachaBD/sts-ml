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

## Plan
- **E1 grid**: 2000 positions; sims {16, 32, …, 1024} × worlds {1, 2, …, 64}, ~25 min. Output: regret vs network requests, split by stakes.
- **E2 confirm**: paired win-rate games for 2–3 frontier settings vs 256 × 8, on validation.
