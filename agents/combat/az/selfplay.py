"""Self-play: B games advanced T decisions per jitted call, resetting finished games from a start bank.

Every decision is recorded with its game's (lane, episode) id; the value target (did that game win) is attached
on the host once the game finishes (replay.py).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

import jaxsim


class Carry(NamedTuple):
    states: object        # jaxsim State [B]
    episode: jax.Array    # [B] int32: per-lane episode counter


class Record(NamedTuple):
    obs: jaxsim.Obs       # [T, B, ...]
    mask: jax.Array       # [T, B, A]
    policy: jax.Array     # [T, B, A] improved policy (search target)
    episode: jax.Array    # [T, B]
    done: jax.Array       # [T, B] game ended by this decision
    won: jax.Array        # [T, B]
    truncated: jax.Array  # [T, B]
    error: jax.Array      # [T, B]
    root_value: jax.Array  # [T, B]


def sample_starts(bank: jaxsim.StartSpec, key, n):
    idx = jax.random.randint(key, (n,), 0, jax.tree.leaves(bank)[0].shape[0])
    return jax.tree.map(lambda x: x[idx], bank)


def make_selfplay(env: jaxsim.Env, policy, bank: jaxsim.StartSpec, games: int, chunk: int, explore: bool):
    init = jax.vmap(env.init)

    def reset(key):
        k1, k2 = jax.random.split(key)
        return init(sample_starts(bank, k1, games), jax.random.split(k2, games))

    def start(key):
        return Carry(states=reset(key), episode=jnp.zeros((games,), jnp.int32))

    def one(params, carry, key):
        k_search, k_step, k_reset = jax.random.split(key, 3)
        s = carry.states
        obs, mask = jax.vmap(env.observe)(s), jax.vmap(env.legal_mask)(s)
        action, weights, root_value = policy(params, s, k_search, explore)
        nxt = jax.vmap(env.step)(s, action, jax.random.split(k_step, games))
        out = jax.vmap(env.outcome)(nxt)
        rec = Record(obs, mask, weights, carry.episode, out.done, out.won, out.truncated, out.error, root_value)
        fresh = reset(k_reset)
        sel = lambda f, x: jnp.where(out.done.reshape((-1,) + (1,) * (x.ndim - 1)), f, x)
        return Carry(jax.tree.map(sel, fresh, nxt), carry.episode + out.done), rec

    @jax.jit
    def run(params, carry, key):
        return jax.lax.scan(lambda c, k: one(params, c, k), carry, jax.random.split(key, chunk))

    return start, run


def make_eval(env: jaxsim.Env, policy, chunk: int):
    """Play one game per start in `bank` to the end (terminal states are absorbing, so finished lanes idle).
    Returns won, truncated, error per start."""
    def one(params, states, key):
        k_search, k_step = jax.random.split(key)
        n = jax.tree.leaves(states)[0].shape[0]
        action, _, _ = policy(params, states, k_search, False)
        return jax.vmap(env.step)(states, action, jax.random.split(k_step, n)), None

    @jax.jit
    def run(params, states, key):
        states, _ = jax.lax.scan(lambda s, k: one(params, s, k), states, jax.random.split(key, chunk))
        return states

    def play(params, bank, key, max_chunks: int = 64):
        n = jax.tree.leaves(bank)[0].shape[0]
        k_init, key = jax.random.split(key)
        states = jax.vmap(env.init)(bank, jax.random.split(k_init, n))
        for _ in range(max_chunks):
            key, k = jax.random.split(key)
            states = run(params, states, k)
            if bool(jax.vmap(env.outcome)(states).done.all()):
                break
        out = jax.device_get(jax.vmap(env.outcome)(states))
        return out

    return play
