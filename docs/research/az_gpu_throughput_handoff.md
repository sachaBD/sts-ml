# Handoff: az (JAX AlphaZero) throughput on the RTX 3060 (2026-10-24)

Goal of the session: get the jaxsim + mctx self-play pipeline (`agents/combat/az`, `apps/az_train`) running on the
local RTX 3060 (12 GB, Windows/WSL), find what limits its throughput, and size runs. No training happened; the
ultimate goal is good play on the Champ fight, so throughput only matters in fights/hour at useful strength.

## State

- Repos up to date. sts_lightspeed `main` carries 8 local commits rebased on the jaxsim work (not pushed).
- `.venv` has the JAX stack (`make jax-env` + `jax[cuda12]==0.11.2`); `jax.devices()` shows the GPU.
- The end-to-end smoke run (`apps/az_train/config/smoke.toml`) passed before the fix. **Not re-run since the fix:**
  `test_az` and the smoke run.
- Commits from this session: `23e8f04` (search fix), `3294d95` (460k comment), `f1dae5a` (bench3060 config),
  `d5d77a2` (probes in `apps/az_bench/probe_*.py`).

## Findings

1. **Fixed: search chains below finished fights (`23e8f04`).** jaxsim allows only END_TURN once a fight is done,
   so `search.py` gave dead states a 100% prior and every revisit grew the tree one node deeper, up to
   num_simulations. mctx walks and backs up all games in lockstep, so the deepest tree in the batch sets the speed
   of every game. Dead states now get a uniform prior. Search values are unchanged (reward and value below a
   finished fight were already 0). 0 errors in the probe runs.
2. **mctx and the stack are fine.** A toy fight with the same search and self-play loop ran at ~4–5 ms per
   simulation step at 512 games, steady over 30 moves (`probe_depth toy`).
3. **More games don't help: the GPU is saturated at 2048.** 4096 games gave +1–4%. Peak memory 2.5–4 GB, so
   memory is not the limit.
4. **Next bottleneck: tree depth, not network arithmetic.** With the small net, trees average depth ~15 (max ~35)
   vs ~6 with the trivial net, and each simulation step costs ~57 ms vs ~22 ms. The small net's own cost is only
   ~2 ms per evaluation. Likely cause: an untrained net's peaked prior pulls the search down long lines. A trained
   net may behave differently.
5. **Hypotheses that turned out wrong:** the network dominates (it doesn't, even with the 460k net); `lax.scan`
   over decisions (`chunk`) causes the slowdown (it doesn't, and early moves just had shallow trees); `observe()`
   is expensive (0.4–0.6 ms at 2048–4096 games).

## Numbers

RTX 3060, 128 simulations, 1 particle, about moves 5–16 of a fight. Each figure comes from 2–3 timed calls of 4
moves, so treat it as roughly ±10% (an estimate; the spread was not measured).

| Network | Games | Simulations/s | Moves/s | Before fix |
|---|---|---|---|---|
| trivial (uniform prior, value 0.5): ceiling | 512 / 2048 / 4096 | 67k / 91k / 95k | 527 / 714 / 743 | ~18–19k |
| small, C++-like (59k, `probe_nets.SmallNet`) | 2048 / 4096 | 36k / 36.5k | 281 / 285 | — |
| reference transformer (460k, `gpu3060.toml`) | 2048 | 33k | 256 | <18k |

Cost of one evaluation (`probe_eval`), 2048 / 4096 games: observe 0.6 / 0.4 ms, legal_mask 0.1 / 0.2 ms, outcome
0.2 / 0.2 ms, small net 1.5 / 2.0 ms, big net 21 / 38 ms (28 / 42 ms with observe and mask).

Comparison with C++. Both rows use ~37 moves per Champ fight, a figure from C++ runs. These are not equal-strength
comparisons:

| | Fights/hour | Search per move |
|---|---|---|
| C++ PV search, 8 workers (`experiments/combat-pv`) | ~14.5k | 800 simulations |
| C++ MCTS teacher, 10 cores | ~1.8k | 20k simulations |
| JAX now, 2048 games (estimate) | ~25–28k | 128 simulations |

## Next steps (suggested order)

1. Re-run `PYTHONPATH=. .venv/bin/python -m agents.combat.az.test_az` and the smoke config with the fix.
2. **Depth cap:** `NET=small MAXD=16 python -m apps.az_bench.probe_depth jaxsim 2048 12` against the uncapped
   run (about 2.5 min). It was interrupted before giving numbers. The C++ notes estimate useful search depth at
   ~1–2 turns (~5–10 moves), so a cap of 16–24 may cost little; if so, add `max_depth` to `SearchConfig`.
   Ceiling if depth is the whole story: about the trivial net's speed (~2.5×).
3. Other depth levers: skip forced moves (one legal action) inside the environment, and fewer simulations
   (Gumbel is designed for 16–32).
4. Size a real run from the probes, then start learning. Particles were 1 throughout. With 1 particle the search
   commits to one sampled future (draws are effectively known inside the tree), so raise particles to ≥4 before a
   serious run.

## Gotchas

- Compilation is ~1–2 min per new shape (games, simulations, network). `apps/az_train` has no persistent compile
  cache (`apps/az_bench/bench.py` does); setting `JAX_COMPILATION_CACHE_DIR` adds one.
- Use `XLA_PYTHON_CLIENT_PREALLOCATE=false` when measuring memory; JAX otherwise reserves 75% of the GPU.
- Run probes with `python -u`: buffered output is lost if the job is killed.
- Measure past the first ~5 moves of a fight. Early trees are shallow, so early-move timings look too good.
