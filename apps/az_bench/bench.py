#!/usr/bin/env python3
"""Throughput benchmark for the JAX self-play stack (agents/combat/az on sts-lightspeed jaxsim).

  PYTHONPATH=. .venv/bin/python -m apps.az_bench.bench            # quick (~30-60 s): the numbers we track
  PYTHONPATH=. .venv/bin/python -m apps.az_bench.bench --full     # more batch sizes and longer timing
  options: [--games 64 256] [--sims 64] [--particles 8] [--json FILE]

Compiled programs are cached on disk (build/jax-cache), so reruns of unchanged code skip compilation.

Three numbers per batch size, on a fixed reference deck bank (seeded random A20 decks of jaxsim-supported cards,
80 HP) and an untrained reference network:
  sim       env.step decisions/s, uniform random legal actions, auto-reset
  search    mctx simulations/s (simulator + network) for one search call over games x particles roots
  selfplay  self-play decisions/s and fights/s at the given search budget (the headline number)
Fight length depends on the policy, so decisions/s is primary and fights/s is reported alongside.
Compile time is excluded (one warm-up call). Results are comparable only at equal settings and jaxsim card set.
On macOS the az package compiles at XLA optimisation level 0 (agents/combat/az/__init__.py), so Mac numbers
understate the CPU; they are a smoke check. Use the GPU box for real numbers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import jaxsim
from agents.combat.az import decks, model, replay, search, selfplay

REFERENCE_DECKS = decks.RandomDecks(count=1024, extra_min=0, extra_max=12, upgrade_prob=0.3, hp_min=80, hp_max=80, seed=0)
REFERENCE_MODEL = model.ModelConfig(d=128, layers=2, heads=4, mlp_ratio=2)   # 460,165 parameters (gpu3060.toml)


def timed(fn, *args, repeat=2):
    jax.block_until_ready(fn(*args))  # compile + warm up
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t0) / repeat, out


def bench_sim(env, bank, games, steps=128):
    init, step = jax.vmap(env.init), jax.vmap(env.step)
    legal, outcome = jax.vmap(env.legal_mask), jax.vmap(env.outcome)

    @jax.jit
    def run(key):
        k0, key = jax.random.split(key)
        s = init(selfplay.sample_starts(bank, k0, games), jax.random.split(k0, games))

        def one(c, k):
            s, fights = c
            ka, ks, kr = jax.random.split(k, 3)
            a = jax.random.categorical(ka, jnp.where(legal(s), 0.0, -jnp.inf))
            s = step(s, a, jax.random.split(ks, games))
            done = outcome(s).done
            fresh = init(selfplay.sample_starts(bank, kr, games), jax.random.split(kr, games))
            s = jax.tree.map(lambda f, x: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), f, x), fresh, s)
            return (s, fights + done.sum()), None

        (_, fights), _ = jax.lax.scan(one, (s, jnp.int32(0)), jax.random.split(key, steps))
        return fights

    dt, fights = timed(run, jax.random.key(0))
    return {"decisions_per_s": games * steps / dt, "fights_per_s": int(fights) / dt}


def bench_search(env, net, params, bank, games, cfg):
    policy = jax.jit(search.make_policy(env, net, cfg), static_argnums=3)
    states = jax.vmap(env.init)(selfplay.sample_starts(bank, jax.random.key(1), games),
                                jax.random.split(jax.random.key(2), games))
    dt, _ = timed(policy, params, states, jax.random.key(3), True)
    return {"simulations_per_s": games * cfg.particles * cfg.simulations / dt, "seconds_per_call": dt}


def bench_selfplay(env, net, params, bank, games, cfg, chunk, repeat):
    start, run = selfplay.make_selfplay(env, search.make_policy(env, net, cfg), bank, games, chunk)
    carry = start(jax.random.key(4))
    s0 = env.init(jax.tree.map(lambda x: x[0], bank), jax.random.key(0))
    buf = replay.empty(env.observe(s0), env.spec.A, games * chunk * (repeat + 1), 1 << 16)
    carry, buf, _ = jax.block_until_ready(run(params, carry, buf, jax.random.key(5)))  # compile + warm up
    t0, fights, decisions = time.perf_counter(), 0, 0
    for i in range(repeat):
        carry, buf, st = jax.block_until_ready(run(params, carry, buf, jax.random.key(6 + i)))
        fights += int(st.games)
        decisions += games * chunk
    dt = time.perf_counter() - t0
    return {"decisions_per_s": decisions / dt, "fights_per_s": fights / dt}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true", help="more batch sizes, longer self-play timing")
    p.add_argument("--games", type=int, nargs="+", help="search/self-play games (quick: 64; full: 64 256)")
    p.add_argument("--sims", type=int, default=32)
    p.add_argument("--particles", type=int, default=1)
    p.add_argument("--chunk", type=int, help="self-play decisions per timed call (quick: 4; full: 16)")
    p.add_argument("--sim-games", type=int, nargs="+", help="simulator batch sizes (quick: 2048; full: 1024 4096)")
    p.add_argument("--json", help="append results as one JSON line")
    a = p.parse_args()
    a.games = a.games or ([64, 256] if a.full else [64])
    a.chunk = a.chunk or (16 if a.full else 4)
    a.sim_games = a.sim_games or ([1024, 4096] if a.full else [2048])
    repeat = 3 if a.full else 2

    cache = Path(__file__).resolve().parents[2] / "build" / "jax-cache"
    jax.config.update("jax_compilation_cache_dir", str(cache))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    t_start = time.perf_counter()
    lap = lambda: f"[{time.perf_counter() - t_start:5.1f}s]"

    spec = jaxsim.DEFAULT_SPEC
    env = jaxsim.make_env(spec)
    bank = decks.random_bank(REFERENCE_DECKS, spec)
    net = model.AZNet(spec, REFERENCE_MODEL)
    s0 = env.init(jax.tree.map(lambda x: x[0], bank), jax.random.key(0))
    params = net.init(jax.random.key(0), env.observe(s0), env.legal_mask(s0))
    cfg = search.SearchConfig(simulations=a.sims, particles=a.particles, max_considered=16, gumbel_scale=1.0)
    result = {"device": str(jax.devices()[0]), "jaxsim_contract": jaxsim.CONTRACT,
              "supported_cards": len(jaxsim.supported_cards()), "sims": a.sims, "particles": a.particles,
              "parameters": int(sum(x.size for x in jax.tree.leaves(params))), "sim": {}, "search": {}, "selfplay": {}}
    print(f"{result['device']}  cards={result['supported_cards']}  params={result['parameters']}  "
          f"search={a.particles} particles x {a.sims} sims")
    for g in a.sim_games:
        r = result["sim"][g] = bench_sim(env, bank, g)
        print(f"{lap()} sim      games {g:5d}: {r['decisions_per_s']:12,.0f} decisions/s {r['fights_per_s']:10,.1f} fights/s")
    for g in a.games:
        r = result["search"][g] = bench_search(env, net, params, bank, g, cfg)
        print(f"{lap()} search   games {g:5d}: {r['simulations_per_s']:12,.0f} simulations/s ({r['seconds_per_call']:.3f} s/call)")
        r = result["selfplay"][g] = bench_selfplay(env, net, params, bank, g, cfg, a.chunk, repeat)
        print(f"{lap()} selfplay games {g:5d}: {r['decisions_per_s']:12,.1f} decisions/s {r['fights_per_s']:10,.2f} fights/s")
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
