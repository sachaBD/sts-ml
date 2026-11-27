# Candidate label: fresh paired grading

- **Scope:** one way to grade selected probe choices, not a requirement for every node. See [training design space](policy.md).

- **Given:** deep, many-particle Gumbel improves on the prior; only a small fraction of moves need it (~4%, prior project result).
- **Goal:** learn *which decisions* benefit. **Not** a rerun of deep search vs. prior.
- **Vocabulary:** [terminology](terminology.md).

```text
sampled decision
  |-- prior ----------------> move A
  `-- probe search        --> move B
                               |
                  A == B? -----+----- A != B?
                     |                   |
                 label = 0        paired comparison
                                  on fresh worlds
```

1. **Sample a decision.**
   - Give every player decision the same small inclusion probability.
   - Include mid-turn decisions. No extra weight after chance nodes.
   - Keep a snapshot for side branches; leave the main fight on its normal self-play rule.
2. **Choose the two moves.**
   - **A:** the move the prior alone would play.
   - **B:** the move the probe selects from the same observation.
3. **If A == B:** record benefit **0**. Stop; no comparison evals.
4. **If A != B:** compare **only A and B**.
   - Sample fresh worlds at the root; do **not** reuse the probe's particles.
   - For each world, clone it twice. Force A in one copy and B in the other.
   - After the forced move, use the **same continuation policy** in both copies, not a forced common action sequence.
   - Reference endpoint: fight end. Score win = 1, loss = 0.
   - Per world: `difference = score(B) - score(A)`.
   - **Benefit label:** mean of the paired differences. Keep negatives.
5. **Train at the sampled root.**
   - **Benefit head:** input = original root observation; target = signed benefit label. No probe output as input.
   - **Value head:** keep the existing sparse main-fight outcome target; do not substitute the paired difference.

**Budget knobs (unset)**

- Probe sampling probability; probe evals / particles; fresh world pairs per disagreement.
- Optional cheaper endpoint: same horizon and value evaluator for both branches; estimates win probability rather than observing the outcome.
- Extra work per sampled decision = probe cost + paired comparison cost on disagreements only.
- **~4% is not the probe sampling probability or necessarily the disagreement rate.**
