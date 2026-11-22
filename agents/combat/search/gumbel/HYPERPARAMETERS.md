# Gumbel search hyperparameters

## Value influence: `value_scale`

`value_scale` controls how strongly search action values influence decisions relative to the policy-network logits. It does not change the backed-up Q estimates.

Completed Q values are used without min-max rescaling (see below):

```text
C = (maxvisit_init + maximum action visits at this node) * value_scale
sigma(a) = C * (completed_Q(a) - min_Q)
improved_policy = softmax(policy_logits + sigma)
```

Root selection additionally uses Gumbel noise and the sequential-halving visit schedule. Interior selection tracks the improved policy using visit counts; it is not simply an argmax of Q.

For a visited action, completed Q is its mean backed-up value. Unvisited actions receive the mixed value described below.

With `maxvisit_init = 50`, maximum action visits of 103, and `value_scale = 0.1`, C is 15.3. A one-percentage-point Q advantage contributes 0.153 logits. A prior logit disadvantage of 2.45 requires a Q advantage of approximately 0.160 to cancel it in the improved-policy scores. This comparison excludes root Gumbel noise.

- Higher scale: stronger search corrections, but greater sensitivity to noisy Q differences.
- Lower scale: greater reliance on the prior, potentially preventing useful corrections and weakening policy-training targets.
- Zero scale: Q contributes nothing to the improved-policy scores.

`maxvisit_init` sets the baseline influence before many visits accumulate. Its current default is 50; `value_scale` defaults to 0.1. Increasing the simulation budget can increase value influence even with both parameters unchanged.

## No local min-max normalization

The current implementation (`sigma()` in `gumbel.cpp`) does **not** min-max normalize completed Q at each node. This differs from mctx's default `qtransform_completed_by_mix_value(rescale_values=True)`, which uses:

```text
normalized_Q(a) = (completed_Q(a) - min_Q) / max(max_Q - min_Q, 1e-8)
```

There is no setting to re-enable it. Subtracting min_Q only adds a common offset to every action's score, so it does not change action preferences; the absolute Q spread is what matters.

Why: normalization discards the magnitude of the local action-value spread (a spread of 0.001 and one of 0.5 both become 1), so small sampling fluctuations can exert strong policy influence. With Q as win probability, keeping absolute differences is meaningful. The cost is that `value_scale` now has to be tuned in win-probability units: too low and search corrections are weak.

## Leaf mixing versus value scaling

The optional guided-rollout leaf evaluator uses:

```text
leaf_value = (1 - rollout_mix) * V_net + rollout_mix * rollout_outcome
```

`rollout_mix` changes the evidence backed up into Q: it trades network prediction against guided-rollout outcomes, with associated bias, variance, and compute costs. `value_scale` changes the influence of the resulting Q estimates on decisions. Both are hyperparameters, but they control different things.

This is also distinct from Gumbel's completion of unvisited action values. In the current implementation:

```text
weighted_Q = prior-weighted mean Q over visited actions
mixed_value = (V_net_at_node + total_visits * weighted_Q) / (total_visits + 1)
```

Here the stored node value may itself come from a blended evaluator. With no visited actions, completion uses that stored node value. This completion formula has no separate mixing setting in the current Gumbel Settings.

## Suggested tuning procedure

1. Freeze the network checkpoint, evaluation cases, leaf mixing, particle count, simulation budget, and Gumbel settings. Change one factor at a time.
2. Try `value_scale` values such as 0.03, 0.1, 0.3, and 1.0. This is a proposed coarse sweep, not an empirically established range. Keep `maxvisit_init` fixed initially.
3. Interpret candidates using `C * delta_Q`: how much win-probability advantage is needed to overcome a typical prior logit disadvantage? This gives an interpretable starting point, not a substitute for evaluation.
4. Compare combat win rate on paired cases/seeds, recording sample size, paired uncertainty, and compute cost. Do not select using the search's own Q estimates alone. Use held-out cases/seeds for final selection.
5. Inspect policy entropy, prior-to-improved-policy changes, action stability across independent search seeds, and whether advantages persist on fresh outcome samples. These are diagnostics, not substitutes for wins.
6. Retest after changing search budget, particle count, leaf mixing, or the network. A good fixed-checkpoint evaluation setting is not necessarily best for learning; validate promising settings in training separately.

Not normalizing does not repair information leakage, strategy fusion, or overfitting to a fixed particle set. Repeated visits are not necessarily independent outcome samples, and the visit multiplier is not a statistical confidence estimate.

## Other search settings

| Setting | Current default | Main effect |
| --- | ---: | --- |
| `simulations` | 256 | Search budget; also affects the visit-dependent value multiplier. |
| `particles` | 8 | Size of the sampled hidden-world pool reused by search. More simulations on the same pool do not create new independent worlds. |
| `considered_actions` | 16 | Root sequential-halving candidate-count setting, capped by legal action count. |
| `gumbel_scale` | 0 | Root perturbation strength; 1 is the conventional exploration setting, 0 removes that perturbation. Particle sampling can still make search stochastic. |
| `sieve_gain` | 0 | Optional progressive sieving (`0` = off). Above `0`, skips decided roots (`min(V_root, 1 - V_root) < 5e-4`) and grows one search in `4×` stages from `16` up to `simulations`, continuing only while `max Q - Q(prior) >= sieve_gain * delta_sims / 1000`. |

Source: `gumbel.hpp`, `gumbel.cpp`, and the guided-rollout leaf evaluator in `../rollout_blend/`. Mctx reference: https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py
