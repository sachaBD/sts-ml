# Model-free RL

**Status: not planned.**

## Idea

Treat run-level decisions (card rewards, possibly more) as an RL problem, e.g. PPO, with real combat inside the environment and reward = act/run success.

## Why not now

It is the least sample-efficient option here:

- **Sparse, delayed reward:** the only signal is the run outcome.
- **Expensive episodes:** every episode costs ~15 real fights (~30 s at the current budget).
- **Nothing reused:** it learns nothing from the dense per-fight labels CARDS exploits.

This is a poor fit for a single machine. It is recorded for completeness.
