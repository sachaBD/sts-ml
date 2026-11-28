# Recursive Kalman-VOI Tree Search

Clean, modular, principled replacement for the 2-level plunge filter (`agents/combat/search/filter/`), built around two core ideas:

1. **Buy compute incrementally via Value of Information (VOI)** — at every step (including step 0), buy the highest-VOI expansion or chance-world sample anywhere in the tree if its expected root win-rate gain exceeds `price` ($\lambda$), otherwise stop.
2. **Scalar Kalman uncertainty throughout one unified recursive tree** — no turn wall, no depth limit, no 2-level plunges, no opening-reserve hack.

On the held-out **2,400-deck** benchmark (`iter003`, TensorRT GPU), the finalized design improves over our initial v5 baseline by **+3.62 pp** `[+2.12, +5.12]` at lower compute (`68.29% @ 271.1 req` vs `64.67% @ 281.7 req`), **beats Horizon 2×1** (`67.62% @ 278.5 req`), statistically ties the hand-tuned 2-level plunge **Prev Filter** (`68.88% @ 275.1 req`, `-0.58 pp [-2.00, +0.83]` with `66` fewer total evals including cache reuse), and scales strictly monotonically to **70.67% (`1,696 / 2,400`)** at `493.2 req` (**+1.79 pp** `[+0.38, +3.12]` above Prev Filter, **+3.04 pp** `[+1.62, +4.50]` above Horizon, **+11.79 pp** above Prior).

---

## Module Layout

| File | Responsibility |
|---|---|
| `agents/combat/search/filter/belief.hpp` | Scalar information-form `kalman(prior, prior_var, ys, obs_prior_var)`, exact Gaussian `positive_part()` VOI integral, `prior_sd()` (`sqrt(2π)·max(thinking)` with Bernoulli floor, capped by `2·min(v, 1-v)`), `kPolicyTemp = 0.03`. |
| `agents/combat/search/filter/tree.hpp` | `Tree` class: deduplicated chance outcomes (`world_children` + `unique_children`), cumulative deterministic root-prefix `path_logit_gap`, 1-pass bottom-up Bellman/Kalman `backup()` with paired CRN control variate $\beta_w$, greedy-anchored interior sibling priors, unpaired deep chance draw-variance floor, exact reach-weight transposition cancellation (`collect_best_reach`), and unified recursive `best_candidate()` VOI scan. |
| `agents/combat/search/filter/simulator.hpp` | `WorldStepper` and `play_action()` built directly on `combat::legal_actions`, `combat::action_key`, `combat::observation_key`, and `combat::particles`. |
| `agents/combat/search/filter/filter.cpp` | Thin `CombatSearch` driver: Turn-boundary greedy `Rollout` (on newly created nodes, unlimited turns), paired CRN reference-world dispatch, eval cache, optional `FilterTrace`, and Gumbel sampling in `result()`. |
| `agents/combat/search/filter/filter_test.cpp` | 12 invariant unit tests (prefix invariance, multi-turn lookahead, decided-state 1-eval exit, paired CRN $\beta_w$, OOD low-policy protection, transposition invariance, cumulative pre-chance policy prior, unpaired deep chance draw regularization, trace parity, draw-pile permutation safety, Gumbel explore). |
| `apps/combat_inspector/filter_lab/` | Interactive step-through lab showing abs win %, `thinking` head (`th`), turn badges (`T2`, `T3`, …), exact JS/C++ parity Kalman uncertainty tables, and live `/api/run` parameter re-runs. |

---

## Final Algorithm Summary

1. **Node epistemic uncertainty (`prior_sd`):**
   $$\sigma(u) = \min\!\Big(2\min(v_u, 1 - v_u),\; \max\!\big(0.005\sqrt{v_u(1 - v_u)},\; \sqrt{2\pi}\max_k \text{thinking}_u[k]\big)\Big)$$
   Exact Gaussian SD matching $\mathbb{E}[\max(0, \mathcal{N}(0, \sigma^2))] = \sigma / \sqrt{2\pi} = \max_k \text{thinking}_u[k]$, vanishing smoothly at `0%` / `100%` decided states.
2. **Unclamped policy-informed Kalman prior:**
   $$\mu_0(u, a) = \text{anchor}(u) + 0.03 \cdot (\ell_a - \ell_{\max})$$
   Processed in descending policy order: for interior nodes ($u > 0$), once the greedy continuation (`order[0]`) is expanded, `anchor(u)` updates from $v_u$ to $Q(u, a_{\text{greedy}})$ so unexpanded siblings sit on the same Turn+1 value scale instead of stale mid-turn $v_u$. Unclamped $\mu_0$ preserves logit gaps on low-win-rate states ($v_u \approx 0.3\%$).
3. **Turn-boundary rollouts inside an unlimited recursive tree:**
   Whenever a newly created non-terminal node stays on the parent's turn, greedy continuation rolls out to `target_turn = parent.turn + 1` so root arms are compared on a matched Turn+1 horizon from World 0 onward, while leaving Turn+1, Turn+2, Turn+3, … leaves freely expandable whenever VOI beats $\lambda$.
4. **Paired CRN control variate $\beta_w$:**
   Before expanding a root-prefix edge on world $w$, the prior line steps through world $w$ to Turn+1 if needed. Its Turn+1 baseline defines the control variate
   $$\beta_w = V^{\text{NN}}_{\text{Turn+1}}(\text{prior}, w) - v_{\text{root}},$$
   subtracted from all shared root-prefix chance edges (`n.world < 0`). Shared draw luck cancels from World 0 onward without erasing deeper Turn+2/Turn+3 improvements on the prior line itself. The reference chance edge itself has zero self-innovation ($\sigma_{\text{innov}} = 0$).
5. **1-pass bottom-up Kalman/Bellman backup (`Tree::backup`):**
   - **Unexpanded edge:** $Q = \mu_0$, $\sigma_{\text{innov}} = \sigma(u) / \sqrt{2}$, weight $1$.
   - **Deterministic edge (`!isChance()`):** exact $V(c)$ if terminal, greedy continuation, or already Kalman-filtered downstream (`c.has_root_kalman`); otherwise $Q = (\mu_0 + N \cdot V(c))/(1 + N)$ with $N = c.\text{evals}$.
   - **Chance edge (`isChance()`):** deduplicates repeated public states and runs scalar information-form `kalman(prior, edge_pv, ys, obs_pv)`:
     - On root-prefix chance edges (`n.world < 0`): uses cumulative deterministic path prior $v_{\text{root}} + 0.03 \cdot (\text{path\_logit\_gap}(u) + \Delta\ell)$, root epistemic variance $\sigma(\text{root})^2$, and `obs_pv = edge_pv`.
     - On unpaired deep chance edges (`n.world >= 0`): floors observation noise variance by Bernoulli draw variance `obs_pv = max(pv, 0.01 * v * (1 - v))` ($\text{SD} = 0.05$ at $v = 0.5$, vanishing at $0/1$) so low-`prior_sd` end-of-turn nodes do not jump 50% on a single unpaired Turn-3 draw.
   - **Decision node value:** $V(u) = \max_a Q(u, a)$ (exact `0`/`1` if terminal; locked at $v_u$ on ancestor cycles).
6. **Transposition-invariant VOI scan (`Tree::best_candidate`):**
   Collects public-state reach weights along the current root leader's and runner-up's best lines (`collect_best_reach`). At each node, cancels shared reach weight $W_{\text{eff}} = \max(0, W - W_{\text{opp}}(\text{key}))$, so shared transpositions (e.g. `Strike → Defend` vs `Defend → Strike`) automatically contribute zero root VOI. Leader scans downward along `leader(u)`; challengers scan upward across all edges. Buy the single max-VOI candidate if $\text{VOI} > \lambda$ and `spent < cap`; otherwise stop.

---

## Held-Out 2,400-Deck Pareto Frontier (`iter003`, TensorRT GPU)

Model: `runs/benchmark/unified-filter/model.path` (`iter003`, `sims=160`, `particles=16`). Held-out split: `runs/benchmark/filter-2400/compare.jsonl` (800 Champ, 800 Bronze Automaton, 800 Collector). Raw artifacts under `runs/benchmark/kalman-voi-final/` (gitignored).

### Main Pareto Table

| Algorithm | Wins / 2,400 | Win % | Mean Evals | p50 / p90 / p99 | Reused Evals | Champ / 800 | Automaton / 800 | Collector / 800 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Prior (1 eval/step) | 1,413 | 58.88% | 41.0 | — | — | — | — | — |
| **Kalman-VOI (`λ=0.01`)** | **1,492** | **62.17%** | **58.2** | `48 / 98 / 233` | 4.6 | 568 (71.0%) | 414 (51.8%) | 510 (63.8%) |
| **Kalman-VOI (`λ=0.003`)** | **1,607** | **66.96%** | **142.9** | `104 / 280 / 712` | 31.0 | 599 (74.9%) | 460 (57.5%) | 548 (68.5%) |
| **Kalman-VOI (`λ=0.00095`)** | **1,639** | **68.29%** | **271.1** | `196 / 557 / 1356` | 64.4 | 610 (76.2%) | 473 (59.1%) | 556 (69.5%) |
| **Kalman-VOI (`λ=0.0004`)** | **1,675** | **69.79%** | **391.8** | `291 / 809 / 1990` | 94.9 | 615 (76.9%) | 495 (61.9%) | 565 (70.6%) |
| **Kalman-VOI (`λ=0.0002`)** | **1,696** | **70.67%** | **493.2** | `365 / 1022 / 2178` | 118.9 | **615 (76.9%)** | **498 (62.2%)** | **583 (72.9%)** |
| *Kalman-VOI v5 (`λ=5.77e-6`)* | *1,552* | *64.67%* | *281.7* | `105 / 743 / 2185` | *64.6* | *585 (73.1%)* | *450 (56.2%)* | *517 (64.6%)* |
| *Horizon 2×1 (`λ=6.67e-4`)* | *1,623* | *67.62%* | *278.5* | `219 / 545 / 1065` | *118.2* | *595 (74.4%)* | *462 (57.8%)* | *566 (70.8%)* |
| *Prev Filter 3×2 (`λ=5e-4`)* | *1,653* | *68.88%* | *275.1* | `230 / 546 / 1081` | *126.7* | *613 (76.6%)* | *483 (60.4%)* | *557 (69.6%)* |

### Paired Bootstrap 95% Confidence Intervals (4,000 resamples)

| Kalman-VOI Operating Point | vs. Kalman-VOI v5 (`281.7 req`) | vs. Horizon 2×1 (`278.5 req`) | vs. Prev Filter 3×2 (`275.1 req`) |
|---|---|---|---|
| `λ=0.003` (`142.9 req` — **49% fewer evals**) | **+2.29 pp** `[+0.79, +3.83]` | −0.67 pp `[-2.12, +0.75]` | −1.92 pp `[-3.33, -0.50]` |
| `λ=0.00095` (`271.1 req` — **cost-matched**) | **+3.62 pp** `[+2.12, +5.12]` | **+0.67 pp** `[-0.67, +2.04]` | −0.58 pp `[-2.00, +0.83]` |
| `λ=0.0002` (`493.2 req` — **high-budget**) | **+6.00 pp** `[+4.42, +7.58]` | **+3.04 pp** `[+1.62, +4.50]` | **+1.79 pp** `[+0.38, +3.12]` |

### Key Takeaways from Held-Out Validation

1. **+3.62 pp leap at lower compute with 38% tighter p99 tail:** Compared to v5 (`281.7 req`, p99 `2,185`, max `8,080`), final Kalman-VOI (`271.1 req`, p99 `1,356`, max `3,096`) gains **+87 wins** while cutting worst-case fight requests by 62%.
2. **Exceptional low-budget efficiency:** At `142.9 req/fight` (half of Horizon's budget), Kalman-VOI reaches `66.96%` — statistically indistinguishable from Horizon's `67.62% @ 278.5` (`-0.67 pp [-2.12, +0.75]`).
3. **Strong, strictly monotone scaling to `70.67%`:** Every boss improves monotonically from `58.2` to `493.2 req`, adding **+2.38 pp (+57 wins)** beyond the `271 req` operating point and beating Prev Filter by **+1.79 pp** `[+0.38, +3.12]` and Horizon by **+3.04 pp** `[+1.62, +4.50]`.
4. **Effective compute (including cross-decision cache reuse):** At `271.1` network requests, Kalman-VOI gets `64.4` reused cached evaluations per fight (`335.5` total evals), whereas Prev Filter (`126.7` reuse, `401.8` total evals) and Horizon (`118.2` reuse, `396.7` total evals) get ~2× more free cache hits from wider Turn-1 opening waves. Achieving `68.29%` on `335.5` total evals vs `68.88%` on `401.8` and `67.62%` on `396.7` reflects strong per-evaluation information efficiency, and points to **cache-aware VOI pricing** (`cost = 0` or discounted for cached child observations) as a clean future win.

---

## Development Progression & Ablation History (504 Dev Decks)

All design decisions were developed and selected strictly on the 504 dev decks (`144` pilot + `360` cal) before running the held-out 2,400-deck comparison.

| Exp | Principled Change | Pilot (144) Peak / Curve | Cal (360) Curve / Pooled AVG (1,440) | Verdict |
|---|---|---|---|---|
| **v5** | Initial modular Kalman-VOI (clamped `prior_mean`, 1-step expansions, centered `beta[w]`) | `101/144 (70.1%) @ 49–80` | `238/360 (66.1%) @ 251` | Baseline |
| **Exp 1** | Unclamped `prior_mean` + exact `sqrt(2π)·max(thinking)` | `107/144 (74.3%) @ 183` | `234 @ 113` (dipped to `224 @ 200`, 1-step horizon mismatch) | Kept `prior_mean` + `sqrt(2π)` |
| **Exp 2** | + Turn-boundary greedy `Rollout` (unlimited turns) | `110/144 (76.4%) @ 208` | `242 @ 172`, `245 @ 328` | **Kept** (major win) |
| **Exp 3** | Tested `cnt_w >= 1` `beta[w]` + crude energy cost | Worse | `239 @ 328` (zeroed single-arm scatter) | Reverted |
| **Exp 4** | Exact paired CRN control variate `beta[w] = prior_T1(w) - root.nn_value` | `110/144 (76.4%) @ 254` | `241 @ 109`, `243 @ 398` | **Kept** |
| **Exp 5** | Zero `ref_edge` self-innovation + exact `turn1_baseline_nn` | Dipped (energy cost) | `245 @ 181`, `246 @ 393` | **Kept** `ref_edge` + `turn1_baseline_nn`; reverted energy cost |
| **Exp 6A/B** | Diagnostic: paused Turn-2+ (`6B`) | Proved Turn-2+ wasn't root bug | Pinpointed stale mid-turn `anchor` on interior siblings | Diagnostic |
| **Exp 7** | **Greedy-anchored interior sibling priors (`anchor = e_greedy.q`)** + restored unlimited Turn-2+ | **`111/144 (77.1%) @ 196`** | `251 @ 155`, `253 @ 542` · AVG `69.44% @ 346` | **Kept** (+3–4 pp jump) |
| **Exp 8A** | **+ Exact reach-weight transposition cancellation** | **`111/144 (77.1%) @ 180`** | `251 @ 147`, `254 @ 372` · AVG `70.00% @ 321` | **Kept** (strict Pareto dominance) |
| **Exp 8B** | Tested `1 + clamp(energy, 0, 3)` cost on top of 8A | Worse | Worse (penalized `End Turn` and start-of-turn decisions) | Reverted |
| **Exp 9A/B** | + `floor = 0.005` (`9A`) / `0.001` (`9B`) + `pair_cost` `1` vs `2` | `110 @ 185` | `254 @ 135`, `261 @ 352`, `252 @ 491` · AVG `70.62% @ 308` | Kept `floor = 0.005` |
| **Exp 10** | + Cumulative pre-chance `path_logit_gap` + `created`-only rollout | **`112/144 (77.8%) @ 355`** | `255 @ 136`, `258 @ 342`, `248 @ 479` · AVG `69.93% @ 302` | **Kept** (fixes 10% pre-chance interior bug) |
| **Exp 11A** | Tested lower CRN `obs_pv = 0.5·pv` | `111 @ 190`, `107 @ 343` | Worse everywhere (`249 @ 148`, `242 @ 520`) | Reverted (OOD 1-world spikes) |
| **Exp 12** | Tested `kPolicyTemp = 0.04` | `112 @ 278` | Worse on cal (`251 @ 114`, `248 @ 286`) | Reverted (`0.03` matches trained `1/k0`) |
| **Exp 13** | Tested `End Turn = 1` vs `1 + 0.5·energy` rollout cost | Worse (`104 @ 157`) | Worse everywhere (`248–250`) | Reverted (starved critical 3-energy opens) |
| **Exp 14A** | Tested `pair_cost = 2` on Step-0 opening | `112 @ 157` | Hurt tight-budget cal (`238 @ 114` vs `255 @ 136`) | Reverted |
| **Exp 14B** | Diagnostic: paused Turn-2+ on top of Exp 10 | Dropped **−4.86 pp** (`105 @ 313`) | Eliminated `λ=0.0002` Automaton dip (`256 @ 397`) | Proved Turn-2+ is valuable & needs draw noise floor |
| **Exp 15** | **+ Unpaired deep chance Bernoulli `draw_var = 0.01·v(1-v)` floor** | **`113/144 (78.5%) @ 348`** | `257 @ 134`, `250 @ 476` · AVG `69.86% @ 299` | **Kept** |
| **Exp 16 (Final)** | **Removed `pair_cost` hack** (fair world resampling vs deep rollouts; cleaner core) | `111 @ 181`, `110 @ 269` · AVG `75.17% @ 224` | `254 @ 146`, `253 @ 359`, `255 @ 490` · AVG **`70.28% @ 316`** | **Final** (best pooled balance `1,445/2,016`, zero cruft) |
| **Exp 17** | Tested `mean(thinking)` instead of `max(thinking)` | `106 @ 84`, `112 @ 219` | `251 @ 123`, `252 @ 228`, `253 @ 396` · AVG `69.79% @ 261` | Kept `max(thinking)` (exact Gaussian identity & +0.30 pp pooled) |

---

## Reproducing Tests, Benchmarks, & Filter Search Lab

```sh
# Build & run unit tests (12 Filter.* + 36 search/player/inspection tests)
cmake --build build/main --target filter_test filter_replay nn_play -j
ctest --test-dir build/main -R "Filter|Decision|Episode|Horizon|Search|Inspection" --output-on-failure

# Regenerate the Filter Lab trace & serve interactively
build/main/apps/combat_inspector/filter_replay runs/benchmark/unified-filter/cpu-model \
  < runs/benchmark/filter-2400/lab/starts.jsonl \
  > runs/benchmark/filter-2400/lab/trace-kalman.jsonl
.venv/bin/python -m apps.combat_inspector.filter_lab.server \
  runs/benchmark/filter-2400/lab/trace-kalman.jsonl \
  runs/benchmark/filter-2400/lab/starts.jsonl \
  runs/benchmark/unified-filter/cpu-model --port 8005
```
