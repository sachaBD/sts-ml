# Probe and ThinkingHead (proposal)

> **Candidate idea, not committed to.** Author: `opus-5-5`, session `0dc13b7c-eb9e-42a1-9df4-7f0f31955be1`.

- **Scope:** fills the *Open* row of [policy.md § Train](policy.md): probe algorithm and budget, grading,
  continuation policy, endpoint, trace retention. Generalises the [paired-grading recipe](readme.md).
  Vocabulary: [terminology](terminology.md); *checkpoint* `b_k` = a request budget, *block* = one search purchase.
- **Stance:** the probe **is** the play search, run past its stop. The head regresses, per budget, the graded
  win gain of whatever move that search commits to. One scalar per budget, no per-action outputs: it is not a
  second policy.

```text
main line ── decision s ──(p_incl)──► snapshot s                     (main fight never waits)
                                        │
   PATH PROBE   play search; λ never stops it; cap drawn at random, independent of s
                trace = per block boundary: (requests, commit, next VOI/request, summary)
                                        │
   commit c_k at checkpoints  16  32  64 128 256 512 …
                               A   A   A   B   B   C          C = distinct commits {A,B,C}
                                        │
                              COUPLED GRADE  fresh worlds, greedy continuation,
                              identical states merged  →  Δ(B), Δ(C);  Δ(A) ≡ 0
                                        │
            gain head (MSE)  ◄── label_k = Δ(c_k)              0 for free when c_k = A
            SieveGate rows   ◄── Δ(commit later) − Δ(commit now)
```

## Why not the #206 head again

| | #206 VoS head | This design |
|---|---|---|
| Label | best Q − Q(prior) on the **same** search: ≥ 0, a max over noisy Qs | signed Δ on **fresh** worlds; exactly 0 when the commit is A |
| What a head can fit | noise level: root Q SD ≈ 3× the top-2 gap (#198), so the label's mean grows with √v(1−v) and #contenders. Consistent with #213: the head's signal was `v(1−v)(1−max π)` | real gain only: label noise has mean 0 |
| Supervision | one row per deep search | one row per probe per checkpoint; rows with c_k = A are free exact zeros |
| Read-out | linear / MLP on CLS | learned queries cross-attending over all tokens |

## 1. Path probe

**One requirement on the play search:** block content depends only on search state, never on λ or the cap;
λ only decides stop. #225 needs one change: run the best VOI-per-request block, instead of every block that
clears λ. Then **play at price λ ≡ the probe trace cut at the first boundary whose next VOI/request < λ.**

```python
CHECKPOINTS = [16, 32, 64, 128, 256, 512, 1024, 2048]   # search requests, cache-blind (traces replay exactly)

def path_probe(s, net, seed):
    cap = draw_cap(seed)                                # P(cap >= 64*2**j) = 2**-j, j = 0..5
    S = PlaySearch(s, net, worlds=stream(seed, "probe"))
    trace = [Boundary(n=0, commit=argmax(net(s).policy))]          # checkpoint 0 = prior move A
    S.run(S.first_wave())                                          # unconditional once entered
    while True:
        blk = max(S.blocks(), key=lambda b: b.voi / b.cost)        # λ-free, cap-free
        trace.append(Boundary(n=S.requests, commit=S.commit(),
                              next_ratio=blk.voi / blk.cost, summary=S.summary()))
        if S.solved() or S.requests + blk.cost > cap:
            return trace, cap                                       # solved: commit holds for all later b_k
        S.run(blk)

commit_at(trace, b) = last(t.commit for t in trace if t.n <= b)
stop_at(trace, λ)   = first(t for t in trace[1:] if t.next_ratio < λ)     # play's (commit, cost) at λ
```

- **Cap law:** equal expected compute per octave; E[cap] ≈ 224 requests; 1 in 32 probes reaches 2048.
  Checkpoint-k rows come from probes with cap ≥ b_k, an unbiased subsample because the cap ⟂ s.
- **Exact offline stop-rule evaluation:** any rule that stops at block boundaries from trace features gets
  its commit and cost from the trace, and that commit is graded (§2). λ, floors and SieveGate are scored on
  fresh-world grading, not on a reference search.
- **Retain:** trace (tens of boundaries × ~30 floats) + graded value sequences. Drop the DAG.
- **Not in v1:** interior-node labels (search-selected states, unequal budgets). Every row is a uniformly
  sampled main-line decision, so the head trains on the play distribution.

## 2. Coupled grade

`C` = distinct commits in the trace (∋ A). `|C| = 1` → every label is exactly 0, no grading.

```python
def grade(s, C, A, seed, W=16, H=4):
    Y = {c: [] for c in C}
    for w in sample_worlds(s, W, stream(seed, "grade")):           # disjoint from the probe's stream
        seqs = run_coupled({c: apply(clone(w), c) for c in C}, pi=greedy_prior, H=H)
        for c in C: Y[c].append(seqs[c])                            # store sequences; η applied in training
    return Y

def run_coupled(branches, pi, H):
    groups = merge_identical(branches)                              # packed full state, incl. RNG
    while any alive group:
        eval one decision state per alive group (batched, shared cache)
        at a turn boundary: append V to each branch in the group
        step with pi; merge groups whose states became identical
        end a group at fight end (append outcome) or after H turn boundaries
    return per-branch value sequences

def ret(seq, η):          # λ-return over turn boundaries; seq[-1] = outcome, or V at H
    m = len(seq)
    return (1 - η) * sum(η**i * seq[i] for i in range(m - 1)) + η**(m - 1) * seq[-1]

Δ[c] = mean_w(ret(Y[c][w]) - ret(Y[A][w]));   Δ[A] ≡ 0
```

- **Continuation = greedy prior.** ~1 eval per decision, and it is Horizon's plunge policy: Δ is measured on
  the line the search itself assumes (used in §4). A stronger continuation is a knob at ~10× grading cost.
- **Coupling.** Byte-identical states have identical futures: merge, and the difference is exactly 0 from there
  at no further cost. Expected on order-only disagreements, the bulk of near-tie flips.
- **Endpoint η.** η = 1, H → fight end: fight outcome, unbiased, noisy. η → 0: next-turn value, quiet, but
  shares V's bias with the probe's own leaves. Turn-boundary V is free (greedy evaluates that state anyway),
  so η is chosen after the fact (§7).

## 3. ThinkingHead

```text
trunk tokens ──► M ≈ 4 learned queries cross-attend over all tokens ──► MLP ──► gain[1..K]
                 (action tokens carry their policy logit, so the head sees A and π's spread)
```

| Output | Label | Rows | Loss |
|---|---|---|---|
| `gain[k]` | `Δ[commit_at(trace, b_k)]`; exactly 0 when that commit is A | every probe with cap ≥ b_k | MSE |

- **Target = the label's conditional mean.** Grading worlds are independent of the probe, so
  E[label_k | s] = expected win gain of spending b_k. MSE fits the best predictor of it from what the trunk
  sees, including "some move beats A here, I can't tell which": the label is the gain of whatever the search
  actually picked, so nothing is split into separately fitted parts.
- **Not a policy:** no per-action outputs or targets. Same π with different stakes → different label.
- **Mostly free:** rows with c_k = A are exact zeros with no grading; only paths that leave A are graded.
- **Signed:** gain[k] < 0 where a shallow search is worse than the prior (cf. #220).
- Read-out cost ≪ one trunk layer. Trunk receives the loss at a small weight (ablation: detached);
  policy/value targets unchanged; head absent ⇒ play is exactly today's.
- gain[K] is policy.md's ThinkingHead scalar (at η = 1, H → end), extended to every checkpoint.

## 4. Consumers at play

| Output | Use | Replaces in #225 |
|---|---|---|
| `gain[k]` at root | floor `b_f = argmax_k gain[k] − λ·b_k`; ≤ 0 → play A, zero evals; else spend ≥ b_f, then VOI stop. Negative gain at small k jumps straight to a deeper floor or to A | `trivial(root)`; VOI stopping early where depth discovery is coming (#217) |
| `gain` at plunge states | branch at the greedy step with the largest gain; EI of deepening e = `g·exp(−gap/g)`, g = max(gain(s_e), 0) | PUCT descent; visit-SE noise model |
| `gain` at root and children | SieveGate features | — |

Second row: a plunge from e follows the prior, so gain(s_e) is the expected upward revision of Q(e) from
thinking at s_e; with an exponential revision of mean g, EI is exactly `g·exp(−gap/g)`.

## 5. Invariants

| Invariant | Holds because | Test |
|---|---|---|
| label_k = 0 when c_k = A, with no grading | same move, same branch | algebra |
| play(λ) = prefix of the probe trace | allocation is λ- and cap-free | play at several λ, compare traces |
| grading worlds ∩ probe worlds = ∅ | separate streams | stream ids |
| after a merge: difference 0, cost 0 | identical state ⇒ identical future | one transposition |
| checkpoint-k rows unbiased | cap drawn from seed before search | — |
| λ-return weights sum to 1; η = 1, H = ∞ ⇒ outcome | algebra | — |

## 6. Cost (requests; cache and coupling only lower it)

```text
per probe = E[cap] + g · W · |C| · H · 5.4          g = fraction of paths that leave A, 5.4 decisions/turn
overhead  = p_incl · per probe / self-play requests per decision
```

Assumed: g = 0.3, |C| ≈ 2.2; W = 16, H = 4; self-play 30 req/dec.
→ ≈ 224 + 228 ≈ **450 per probe**; p_incl = 1% ⇒ **≈ +15% self-play compute**, ~10k probe rows (≈ 3k graded)
per 1M decisions (one 20k-fight iteration).

## 7. First checks (cheap → expensive)

1. **Label reliability, no training.** 1k probes with |C| > 1, graded twice on disjoint streams: split-half
   correlation of Δ̂ for η ∈ {0, .5, .8, 1}. Pick (W, η). Below ~0.3 at W = 16 → raise W first. ≈ 1.7M requests.
2. **Head, offline.** Train on one iteration's rows; hold out a later iteration on disjoint decks. Metric: share
   of total graded gain in the top 10% by gain[K]; labels are unbiased, so no reference-search bias. Baseline:
   `v(1−v)(1−max π)`. Kill the head if it doesn't clearly beat it; probe + grader still pay off as SieveGate rows
   and exact stop-rule evaluation.
3. **Fights.** #225 + §4 consumers vs #225's λ-frontier: 2000 validation fights, deck-paired, equal evals/fight.

## Stretch

- **Active deep probing:** draw the cap from the head, P(cap ≥ b_k | s) rising with gain[k] / (λ·b_k), floored;
  weight checkpoint-k rows by 1 / P(cap ≥ b_k | s). 2048-request probes go where depth should pay; still unbiased.
- **Verified deep moves as policy targets:** Δ[c_K] − 2·se > 0 is a fresh-world-verified improvement over A.
  A separate, opt-in teacher for the policy head; never pooled silently.

## Knobs (defaults)

| Knob | Default |
|---|---|
| `p_incl` | 1%, uniform over player decisions |
| checkpoints | 16 … 2048, ×2 |
| cap law | P(cap ≥ 64·2^j) = 2^−j |
| W, H, η | 16 worlds, 4 turn boundaries, η from check 1 |
| continuation | greedy prior |
| loss | MSE (Huber would shrink the heavy tail the head exists to find) |
| thinking loss weight | small (e.g. 0.1), trunk not detached |
