# Literature: ideas behind counterfactual rescue

This is a conceptual reading map from existing knowledge, not a freshly verified
literature survey. These methods motivate mechanisms; none establishes that this
Slay the Spire implementation will work. Our proposal is a combination, not an exact
reproduction of any paper.

| Approach | Relevant lesson | Application here | Important limitation |
|---|---|---|---|
| DAgger / interactive imitation learning | Query the expert at states the learner actually visits, rather than only copying expert trajectories. | Teacher proposals at learner-path disagreements. | Expert actions need not be optimal; ordinary DAgger is not outcome-based action-value estimation. |
| Go-Explore | Return to promising states and explore from there instead of repeatedly rediscovering the route. | Revisit recoverable positions and deliberately explore alternatives. | Successful trajectories need robustification; privileged resets are a training tool, not an evaluation capability. |
| Reverse curriculum generation | Start near successful completion and progressively move the starting distribution backward. | Learn late teacher continuations, then earlier parts of the fight. | Success from late resets may not transfer to reaching those states from the real start. |
| Gumbel AlphaZero / sequential halving | Explicit root-action allocation gives alternatives a meaningful chance under finite search budgets. | Guarantee evaluation of neglected legal actions before allowing prior-driven allocation. | It does not repair a biased leaf evaluator by itself; completed continuation evidence is a separate intervention. |
| Expert Iteration | Alternate search-based policy improvement and learning to amortize that improvement. | Require stronger search targets to yield a stronger trained agent over successive cycles. | The improvement step must actually be strong enough; self-generated targets need not improve the policy. |

## Reference starting points

- Ross, Gordon & Bagnell (2011), *A Reduction of Imitation Learning and Structured
  Prediction to No-Regret Online Learning* (DAgger).
- Ecoffet et al. (2021), *First return, then explore* (Go-Explore).
- Florensa et al. (2017), *Reverse Curriculum Generation for Reinforcement Learning*.
- Danihelka et al. (2022), *Policy improvement by planning with Gumbel*.
- Anthony, Tian & Barber (2017), *Thinking Fast and Slow with Deep Learning and Tree Search*
  (Expert Iteration).

## Our specific distinction

The old correction experiment supplied whole teacher trajectories as a small replay
fraction. The proposed rescue recipe instead establishes a learnable continuation
boundary, guarantees alternative-action exploration where needed, and trains on explicit
corrective evidence. It retains failures and evaluates complete fights without resets.

Do not confuse:
- teacher continuation value with the learner's own continuation value;
- a winning replay with proof an action is best;
- increased prior probability with improved searched play;
- matched root particles with identical future random events after actions diverge;
- accurate fitting at privileged restarts with improved autonomous full-fight performance.
