"""Cost of one search evaluation at B games: observe, legal_mask, outcome, step, network on precomputed obs, full evaluate.

  PYTHONPATH=. .venv/bin/python -u -m apps.az_bench.probe_eval 2048 4096
Caveat: `step` uses a constant END_TURN action, which XLA partly constant-folds; don't trust its number.
"""
import sys
import time

import jax
import jax.numpy as jnp

import jaxsim
from agents.combat.az import decks, model, search, selfplay
from apps.az_bench.bench import REFERENCE_DECKS, REFERENCE_MODEL

from apps.az_bench.probe_nets import SmallNet, TrivialNet

spec = jaxsim.DEFAULT_SPEC
env = jaxsim.make_env(spec)
bank = decks.random_bank(REFERENCE_DECKS, spec)
t0 = time.perf_counter()


def timed(name, B, f, *args, n=20):
    jax.block_until_ready(f(*args))
    tr = time.perf_counter()
    for _ in range(n):
        out = f(*args)
    jax.block_until_ready(out)
    dt = (time.perf_counter() - tr) / n
    print(f"[{time.perf_counter() - t0:5.1f}s] B={B:5d} {name:28s} {dt * 1000:7.2f} ms", flush=True)


for B in (int(x) for x in sys.argv[1:]):
    states = jax.vmap(env.init)(selfplay.sample_starts(bank, jax.random.key(1), B), jax.random.split(jax.random.key(2), B))
    # advance a few random moves so states are mid-fight
    step = jax.jit(lambda s, k: jax.vmap(env.step)(s, jax.random.categorical(
        k, jnp.where(jax.vmap(env.legal_mask)(s), 0.0, -jnp.inf)), jax.random.split(k, B)))
    for i in range(6):
        states = step(states, jax.random.key(10 + i))
    obs = jax.jit(jax.vmap(env.observe))(states)
    mask = jax.jit(jax.vmap(env.legal_mask))(states)
    timed("observe", B, jax.jit(jax.vmap(env.observe)), states)
    timed("legal_mask", B, jax.jit(jax.vmap(env.legal_mask)), states)
    timed("outcome", B, jax.jit(jax.vmap(env.outcome)), states)
    timed("step", B, jax.jit(lambda s, k: jax.vmap(env.step)(s, jnp.full((B,), spec.END_TURN), jax.random.split(k, B))),
          states, jax.random.key(0))
    for name, net in (("small", SmallNet(spec)), ("big", model.AZNet(spec, REFERENCE_MODEL))):
        p = net.init(jax.random.key(0), jax.tree.map(lambda x: x[0], obs), mask[0])
        timed(f"net {name} (obs given)", B, jax.jit(jax.vmap(lambda o, m: net.apply(p, o, m))), obs, mask)
        ev = search.make_evaluate(env, net)
        timed(f"evaluate {name} (all)", B, jax.jit(lambda s: ev(p, s)), states)
    ev = search.make_evaluate(env, TrivialNet(spec.A))
    timed("evaluate trivial (all)", B, jax.jit(lambda s: ev({}, s)), states)
