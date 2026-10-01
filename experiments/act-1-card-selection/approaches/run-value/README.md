# Run value (afterstate value learning)

**Status: alternative; likely needed later as CARDS' cutoff value.**

## Idea

Play many real runs with exploration in card picks. Learn V(run state after the choice) = P(success) from how runs end (Monte Carlo or TD, TD-Gammon style). At a reward, pick the choice whose post-choice state has the highest V.

## Trade-offs

- **For:** simple. There is no combat approximator, so no approximator bias. The label is the real objective.
- **Against:** one outcome label per run, spread over ~8 picks. Credit assignment is slow and data hungry. It also doesn't use search or compute at decision time: all knowledge must be learned in advance.

## Role

- **Standalone:** a fallback if a combat approximator cannot resolve per-card differences.
- **Inside CARDS:** the value at the rollout cutoff when the horizon extends beyond what rollouts can reach cheaply.
