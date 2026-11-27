# Candidate Design — Belief-DAG VOI Search (`Sift-DAG`)

*(See [spec.md § 0](spec.md) for vocabulary.)*

## 1. Core Idea
Treat the root NN evaluation (`1 eval`) as **state $t = 0$ of the public-observation Directed Acyclic Graph (DAG)**, and drive the entire search with **one loop** and **one currency** (Value of Information, **VOI** = expected root win-rate gain, divided by new NN evals):

- At every step (including $t = 0$), score candidate compute blocks $B$ by efficiency $\rho(B) = \text{VOI}_{\text{root}}(B) / \text{Cost}(B)$.
- Stop as soon as $\max_B \rho(B) \le \lambda$.
- Separate the **two orthogonal reasons** a challenger $e$ could beat current leader $\hat{a}$:
  1. **Aleatoric (draw / roll noise across particles):** lucky/unlucky draws or RNG rolls $\implies$ buy **`PARTICLES`** on **shared particles** (evaluating contenders on the exact same `particles[p]` indices).
  2. **Epistemic (greedy continuation error):** unsearched actions at root or along a mid-turn greedy plunge $\implies$ buy **`OPEN`**, **`WIDEN`**, or **`BRANCH`** priced from node `thinking` heads plus an interior prior uncertainty floor.

> **Prefix invariance:** for $\text{Cost}(B) > 0$, $\max_B (\text{VOI}(B)/\text{Cost}(B)) \le \lambda \iff \max_B (\text{VOI}(B) - \lambda\,\text{Cost}(B)) \le 0$. Block ordering (`argmax VOI/Cost`) is **independent of $\lambda$**, so play at price $\lambda > 0$ stops on an exact prefix of a $\lambda = 0$ training probe.

---

## 2. Lifecycle & Emergent $t = 0$ Exit

```text
t = 0 (1 eval spent: root only)                 t > 0 (public DAG across shared particles)
───────────────────────────────                 ──────────────────────────────────────────
      [ root r ]                                              [ root r ]
  V0, logits, thinking[0..3]                          e0 (hat_a) /    |    \ e2 (unvisited)
  N_e = 0 for all e                                             /     | e1  \
  leader hat_a = argmax S(e) = prior A                [u1] (transp.) [u2]   (WIDEN?)
                                                        \           /  \
  Only candidate: OPEN(r)                                \         /    \ (greedy plunge)
  rho = max_k g_r[k] / c_k                                `-> [u_shared] [u_hot: tau > 0]
                                                                          |  (BRANCH!)
  rho <= lambda? ──► STOP (1 eval!)                                   [Turn+1 leaf z]
  rho >  lambda? ──► run Wave 1                                           |  (PARTICLES)
```

---

## 3. DAG State, Scoring & Root Sampling

### Node Readout (on expansion of public state $u$ with $K_u$ actions)
- Checkpoint gains: $g_u[k] = \max(0, \text{thinking}_u[k])$ at fixed eval budgets $c = [8, 24, 64, 160]$.
- **Interior unrealized thinking potential $\tau(u)$:** expected upward value revision at $u$ from branching off its greedy move $g_u = \arg\max \text{logits}_u$:
  - Deterministic prior uncertainty floor (built from `s0 = prior_se`, `0` extra params; enables cold-start search when `thinking == 0` and recovers from head blind spots during $\lambda = 0$ probes):
    $$\tau_{\text{floor}}(u) = \tfrac{1}{2} s_0 \sqrt{v_u(1 - v_u)}\big(1 - \pi_u[g_u]\big)$$
  - Initial potential: $\tau_0(u) = \max\!\big(\max_k g_u[k],\ \tau_{\text{floor}}(u)\big)$.
  - Remaining after visiting edge set $E_u$ with realized gain $\hat{\delta}_u = \max(0, V_{\text{backed}}(u) - Q_u(g_u))$:
    $$\tau(u) = \max\big(0,\ \tau_0(u) - \hat{\delta}_u\big) \cdot \left(1 - \sum_{i \in E_u} \pi_u[i]\right)$$
  - Automatically `0` when $K_u \le 1$, when $\pi_u[g_u] \approx 1$, or once $u$'s alternatives are explored.

### Root Action Scoring & Gumbel Action Selection
- Fit additive particle difficulty $v_{e,p} \approx \mu_e + \beta_p$ ($\sum_p \beta_p = 0$) across all `(root_edge e, particle p)` evaluations.
- Prior-regularized root score (valid at **both** $N_e = 0$ and $N_e > 0$):
  $$S(e) = \hat{Q}(e) + \frac{\ell_e - \ell_{\max}}{k_0 + N_e}, \qquad \hat{Q}(e) = V_0(r) \text{ when } N_e = 0$$
- Search leader during tree expansion: $\hat{a} = \arg\max_e S(e)$; score gap: $\text{gap}(e) = \max(0, S(\hat{a}) - S(e))$.
- **Improved logits & `gumbel_scale` root selection (`result()`):**
  $$z_e = \ell_{\max} + k_0 \big(S(e) - V_0(r)\big)$$
  - Policy target reported by `result()`: noise-free `softmax(z)`.
  - Played action: $\arg\max_e \big(z_e + \text{gumbel\_scale} \cdot G_e\big)$ with $G_e \sim \text{Gumbel}(0, 1)$:
    - When $N_e = 0$ (`1-eval` exit): $z_e \equiv \ell_e$, so `gumbel_scale = 1` performs exact softmax sampling from the raw policy prior.
    - When $N_e > 0$ (searched): $z_e$ is strictly monotone in $S(e)$, so `gumbel_scale = 1` samples from `softmax(z)` in self-play while `gumbel_scale = 0` deterministically commits $\arg\max_e S(e)$ in evaluation and probes.

---

## 4. Transposition Cancellation & Root Relevance

```text
          [ root ]
      hat_a /  \ e          Both hat_a and e transpose into u_shared (omega = 1.0 each),
           /    \           while only e reaches u_only (omega_e = 1.0, omega_hat_a = 0.0).
      [u_shared] [u_only]
                            delta_u(e, u_shared) = max(0, 1.0 - 1.0) = 0   ──► VOI = 0 (skip!)
                            delta_u(e, u_only)   = max(0, 1.0 - 0.0) = 1.0 ──► VOI > 0 (branch!)
```

- Track $\omega_e(u) \in [0, 1]$: fraction of root edge $e$'s evaluated particles whose current best Turn-0 path passes through interior node $u$.
- **Differential reachability:** $\delta^+_u(e) = \max(0,\ \omega_e(u) - \omega_{\hat{a}}(u))$.
- Shared transpositions shift $Q(e)$ and $Q(\hat{a})$ equally ($\delta^+_u(e) = 0$), so neither `thinking(u)` nor $\tau_{\text{floor}}(u)$ wastes evals on them during the root decision.

---

## 5. Candidate Compute Blocks & Unified Loop

Let $n$ = search NN evals spent after root, and $\hat{\Delta}_{\text{realized}}(n) = \max(0, \hat{Q}(\hat{a}) - \hat{Q}(A))$.

| Block | Available when | What it buys (batched wave) | Scale / VOI formula | Expected `Cost` (evals) |
|---|---|---|---|---|
| **`OPEN` / Reserve** | $t = 0$ (or $n < c_k$) | At $t=0$: plunge top-$m_0$ plausible contenders on $p_0=2$ shared particles ($m_0 \in [2..4]$). At $n>0$: non-myopic reserve keeping local allocation alive | $\text{VOI}_k = \max\big(0,\ g_r[k] - \hat{\Delta}_{\text{realized}}(n)\big)$ (when $\lambda = 0$ and $K_r \ge 2$, Wave 1 opens unconditionally) | $c_k - n$ |
| **`PARTICLES(e)`** | $N_{\hat{a}}, N_e > 0$ and chance seen | Plunge $\{\hat{a}, e\}$ on $p_{\text{wave}}$ new shared particles (+ Turn+$H$ rollout on Turn+1 leaves) | $\text{EI}_{\text{Laplace}}(\text{gap}(e),\ \sigma_{\text{part}}(\hat{a}, e, p_{\text{wave}}))$ from paired diffs $v_{e,p} - v_{\hat{a},p}$ | $2 \cdot p_{\text{wave}} \cdot \bar{c}_{\text{lane}}(\hat{a}, e)$ |
| **`WIDEN(e)`** | $N_e = 0$ (best unvisited $e$) | Plunge $e$ alone on $\hat{a}$'s **existing** $p_0$ particles (instant paired comparison at half wave cost!) | $\text{EI}_{\text{Laplace}}\!\left(\frac{\ell_{\max} - \ell_e}{k_0},\ \sqrt{s_0^2 V_0(1-V_0) + \tau_0(r)^2}\right)$ | $p_0 \cdot \bar{c}_{\text{lane}}$ |
| **`BRANCH(e, u)`** | Interior Turn-0 $u$ with $\tau(u) > 0$, $\delta^+_u(e) > 0$ | Expand next best unvisited prior action(s) at $u$ across particles where $e$ reaches $u$, plunging to Turn+1 | $\sigma \cdot \exp(-\text{gap}(e) / \sigma)$ where $\sigma = \delta^+_u(e) \cdot \tau(u)$ | Uncached leaves across $e$'s particles at $u$ |

### Single Control Loop

```python
def advance(dag, lambda_price):
    while evals_spent < max_evals:
        if visited_root_edges == 0:
            if K_root <= 1:
                return commit_with_gumbel()        # forced action: 1 eval
            rho_open = max(g_root[k] / c[k] for k in range(4))
            if lambda_price > 0 and rho_open <= lambda_price:
                return commit_with_gumbel()        # 1-eval emergent exit!
            return queue_opening_wave()            # lambda == 0 always opens Wave 1

        best_block = max(local_blocks(dag), key=lambda B: B.voi / B.cost, default=None)
        if best_block is None:
            return commit_with_gumbel()

        rho_local   = best_block.voi / best_block.cost
        rho_reserve = max(((g_root[k] - delta_realized) / (c[k] - n)
                           for k in range(4) if c[k] > n), default=0.0)

        if lambda_price > 0 and max(rho_local, rho_reserve) <= lambda_price:
            return commit_with_gumbel()            # global stop!
        return queue_wave(best_block)              # spend locally on highest VOI/eval!
```

---

## 6. Operating Modes & Parameters

| Role | `lambda` | `gumbel_scale` | `max_evals` | Behaviour |
|---|---|---|---|---|
| **Evaluation** | `> 0` (e.g. `1e-4`) | `0.0` | `512` | Deterministic best move; 1-eval exit on ~96% of states |
| **Self-Play Fight** | `> 0` (or `∞` for pure prior) | `1.0` | `512` | Gumbel-samples root action from `softmax(z)` for trajectory diversity |
| **Training Probe** | `0.0` | `0.0` | `160` | Deterministic full trace; commits at `[8, 24, 64, 160]` graded on `8` fresh shared particles (`horizon_turns = 2`) |

### Parameters
- **Deployment / Mode Dials (`2`):** `lambda` ($\lambda$), `gumbel_scale` (`0.0` or `1.0`).
- **Belief Priors (`2`):** `readout_k0` (~30), `prior_se` (~0.05 — sets both particle-diff prior and $\tau_{\text{floor}}$).
- **Wave Shapes (`2`):** `wave_particles` (2–4), `horizon_turns` (2).
- **Guardrails:** `max_evals`, `particles` pool cap (16).
