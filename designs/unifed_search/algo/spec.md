# Unified Combat Search — Scope & Specification

## 0. Vocabulary
| Term | Plain Meaning |
|---|---|
| **Eval** | One neural-network forward pass on one state (`value`, `logits`, `thinking`). A batch of 32 states = 32 evals. |
| **Particle** | One concrete sample of hidden state (unknown draw-pile order) and future RNG stream consistent with what the player currently sees (`combat::particles(...)`). |
| **Shared / Paired Particles** | Evaluating two or more candidate moves on the **exact same particle indices** (`particles[0]`, `particles[1]`, …) so identical draw order and RNG rolls hit both moves equally and draw luck cancels out in `Q_p(move B) - Q_p(move A)`. |
| **Public DAG (Directed Acyclic Graph)** | Search graph keyed by order-invariant public observation (`observation_key`), so different action orders that reach the same visible state merge into one shared node instead of duplicating subtrees. |
| **Transposition** | Two different action sequences (e.g. *Strike → Defend* vs. *Defend → Strike*) landing on the exact same public state. |
| **Plunge** | Stepping forward through mid-turn decisions using the network's fast argmax policy prior (`logits`) until reaching the next turn boundary (`Turn+1`) or fight end. |
| **VOI (Value of Information)** | Expected increase in fight win probability of our **committed root move** gained by buying a block of extra search compute. |
| **Aleatoric vs. Epistemic uncertainty** | **Aleatoric** = randomness across particles from card draws / enemy rolls (shrunk by testing more shared particles). **Epistemic** = error from assuming greedy prior play at unbranched interior choices (shrunk by branching the tree deeper/wider). |

---

## 1. Objective
Design **one classical combat Public-DAG search algorithm** that serves evaluation play, self-play trajectory collection, and `ThinkingHead` training probes:

- **Maximise fight win rate at any fixed average NN-eval budget per fight** (Pareto frontier of win rate vs. actual `network_requests`), controlled by a single compute-price dial $\lambda$ (expected root win-rate gain required per NN eval).
- **Single loop, emergent entry:** the 1-eval "play policy prior immediately" fast path must fall out as iteration 0 of the search loop itself—no separate pre-gate or heuristic skip rules.
- **No auxiliary learned gate:** search control uses classical Value of Information (VOI) math over the DAG and NN head outputs, with no separate CPU tree model to co-train.
- **Self-sufficient across all three training/play roles:**
  - **Evaluation:** deterministic root commitment at price $\lambda > 0$.
  - **Self-play:** stochastic root action sampling (`gumbel_scale`) across fights so non-probe trajectories explore diverse states, on both 1-eval exits and searched roots.
  - **Training probes:** deterministic uncapped search ($\lambda = 0$, up to `max_evals`) with enough intrinsic interior optimism to discover improvements even from an untrained or blind `ThinkingHead`.

---

## 2. Hard Constraints & Runtime Contract
- **Asynchronous batched interface (`stsrl::search::CombatSearch`):**
  - Search never calls the network directly; it emits waves of `Query{state, legal_actions}` via `pending()` and receives `evaluated(evaluations)` from batched GPU/CPU workers.
  - Waves should batch multiple independent leaf queries where possible to amortize GPU dispatch latency.
- **Cost accounting:**
  - Price **actual new NN evaluations**, not simulator steps or trajectories.
  - Forced single-action states (`legal_actions.size() == 1`), terminal states, and already-evaluated public DAG nodes cost `0` NN evals.
- **Cross-decision cache compatibility (`player::Answers`):**
  - Evaluations keyed by encoded token request carry forward to the next decision in the fight at `0` GPU requests when that public state is reached on the played line.

---

## 3. Available Data at Play Time

### A. Per-State Neural Network Outputs (1 trunk pass = 1 `eval`)

```text
State + K legal actions ──► Shared Transformer Trunk ──┬──► value        : float in [0, 1]
                                                       ├──► logits[0..K-1]: float[K]
                                                       └──► thinking[0..3]: float[4] (signed)
```

- **`value` $\in [0, 1]$:** estimated fight win probability from this state.
- **`logits[0..K-1]`:** unnormalized prior log-probabilities over deduplicated legal actions; `argmax(logits)` is the 1-eval prior move $A$.
- **`thinking[0..3]` (signed floats):**
  - Predicts $\mathbb{E}[\text{win}(\text{commit at } c_k) - \text{win}(A)]$ at fixed cumulative search NN-eval budgets $c = [8, 24, 64, 160]$ (`0` when `commit == A`, graded on fresh shared particles when `commit != A`).
- **Latent in `model.pt` (not currently exported to ONNX, available at `0` trunk cost if needed later):**
  - `outcome[0..19]`: 10 loss bins (by enemy HP left) + 10 win bins (by player HP fraction left).
  - `auxiliary[0..2]`: `hp_lost`, `enemy_hp_lost` (by next player turn), `turns_remaining`.

### B. Simulator & Public-Graph Primitives (`environments/combat/public.hpp`)
- **`observation_key(state)`:** order-invariant public hash (hand, discard, exhaust pile permutations collapse to the same node).
- **`legal_actions(state)` & `action_key(state, action)`:** canonical deduplicated actions (identical cards/potions collapse to one edge).
- **`particles(observed, count, stream)`:** samples particles $p$ consistent with public state (hidden draw-pile order + deterministic per-particle RNG stream so contenders can share identical particles).
- **`action.execute(state).isChance()`:** exact flag indicating whether a step consumed hidden state / RNG (draw, shuffle, random target, generation) vs. transitioned deterministically.
- **`state.turn` & `state.outcome`:** exact turn boundaries and terminal win/loss (`PLAYER_VICTORY` = `1.0`, loss/turn-cap = `0.0`).

---

## 4. Required Behaviours of Any Candidate Solution
- **Zero-search exit (`1 eval`):** finishes in 1 eval when expected gain per eval at $t = 0$ is $\le \lambda$ (including $K \le 1$ and decided states).
- **Non-myopic persistence:** does not quit after a flat shallow wave when `thinking(root)` predicts positive net payoff at a deeper checkpoint that has not yet been realized or refuted.
- **Shared-particle variance reduction:** compares root contenders on the same particles and adjusts for particle difficulty so lucky draws do not masquerade as strong moves.
- **Targeted interior resolution & blind-spot recovery:** branches interior states where `thinking(child)` (or prior action uncertainty) is high and root-relevant, while skipping shared transpositions.
- **Clean exploration separation:** keeps deterministic search trajectories aligned between Play ($\lambda > 0$) and Probe ($\lambda = 0$), while supporting stochastic root move sampling (`gumbel_scale > 0`) for self-play fight diversity.
