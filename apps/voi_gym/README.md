# Measurement VOI gym

No game engine, model weights, HTTP server, or build needed. From the repository root:

```sh
PY=.venv/bin/python
$PY -m unittest agents.combat.search.voi_gym.test_model apps.voi_gym.test_session
$PY -m apps.voi_gym --seeds 1024 --budget 16 --prices .005 .002 .0005
```

```
choose opener (Strike / Bash / Defend)
  -> observe enemy intent (attack / brace)
    -> choose response (Strike / Defend)
      -> duel resolves with a fixed, hidden matchup-specific win probability
```

Search happens **before** playing. The recommendation is a contingent plan: one opener and one response per observable intent. Improving a response is valuable even if the opener never changes. No further search is assumed after committing to the plan.

### GUI contract (in-process Python)

```python
from apps.voi_gym.session import Session
s = Session(seed=42, budget=16, price=.002)
s.snapshot()          # seed, budget/spend, plan, value, root Qs, leaf means/SDs,
                      # chance probability, uncertainty regime weights, history
s.candidates()        # sorted measurements: ID, linear terms, gain, cost, net, affordable
s.step()              # take best positive-net affordable measurement; None = play
s.step(candidate_id)  # manual counterfactual; respects budget, bypasses price
s.grade()             # optional oracle overlay ONLY: hidden true plan win probabilities
```

- Leaf IDs are `opener/intent/response`. A row observes the weighted sum of its named leaves, not each term individually.
- All results are JSON-compatible. For rewind/fork, recreate the session with the same config and replay history IDs through `step(id)`. Repeated measurements get fresh noise; alternative search orders use the same per-ID stream.
- History records the predicted observation and its predictive SD **before** seeing the sample. Leaf SD is epistemic uncertainty, not observation SD. Gaussian approximations can lie outside `[0,1]`; do not silently clamp them in the GUI.

### What the model claims

- A latent vector holds all leaf win probabilities. A scout returns a noisy value estimate, not a binary win. A paired scout returns a lower-noise response difference (shared noise cancels), at twice the cost. Both are synthetic measurement contracts.
- The prior and observation likelihood have two possible common scales. In each regime the Kalman update is exact; the shared scaling gives the same mean update in both. Bayes' rule updates regime probabilities using the observation. A surprising residual can increase the next measurement's VOI; it does **not** automatically trigger a repeat or erase the evidence.
- For each feasible plan, its value after one measurement is a line in the scalar innovation. The upper envelope is the new expectimax root value. Integrate it under the predictive mixture and subtract the current root value. This is exact **one-step** VOI under the assumed model, not optimal multi-step planning.
- Stop is a candidate worth zero. Rank by `expected gain - price * cost` subject to the hard remaining budget. Cost is scout units, not Python runtime.
- `grade()` evaluates the chosen plan against hidden probabilities (no extra Bernoulli sampling error). The no-search comparator uses the same prior; the oracle knows the probabilities, **not** the future enemy intent.

### Limits before transfer

- Full tiny tree and explicit contingent plans: an exact reference, not scalable policy enumeration for real combat.
- The gym samples from the model's scale family but clips true win probabilities. It tests decision accounting, not real NN calibration. Global scale uncertainty is not a model of a single corrupt scout, NN bias, or a repeatedly queried deterministic evaluator.
- `--fixed-scale` disables adaptation while matching initial predictive variance. Compare costs as well as gains: a shared price is not matched compute.
