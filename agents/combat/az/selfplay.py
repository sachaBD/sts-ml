"""Self-play on the device: B games advanced T decisions per jitted call, writing straight into the replay buffer.

Finished games restart from a random deck of the bank in the same decision (lane recycling), so no lane idles.
Game id = episode * B + lane, unique per run; the replay buffer keys value targets by it.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

import jaxsim
from agents.combat.az import replay


class Carry(NamedTuple):
    states: object         # jaxsim State [B]
    episode: jax.Array     # [B] int32 per-lane game counter


class Stats(NamedTuple):   # sums over a chunk
    decisions: jax.Array
    games: jax.Array
    wins: jax.Array
    truncated: jax.Array
    errors: jax.Array
    win_hp: jax.Array      # sum of final hp over wins


def sample_starts(bank: jaxsim.StartSpec, key, n):
    idx = jax.random.randint(key, (n,), 0, jax.tree.leaves(bank)[0].shape[0])
    return jax.tree.map(lambda x: x[idx], bank)


def make_selfplay(env: jaxsim.Env, policy, bank: jaxsim.StartSpec, games: int, chunk: int):
    """Returns start(key) -> Carry and run(params, carry, buffer, key) -> (carry, buffer, Stats).
    run donates carry and buffer (updated in place)."""
    init = jax.vmap(env.init)
    lanes = jnp.arange(games, dtype=jnp.int32)

    def reset(key):
        k1, k2 = jax.random.split(key)
        return init(sample_starts(bank, k1, games), jax.random.split(k2, games))

    def start(key):
        return Carry(states=reset(key), episode=jnp.zeros((games,), jnp.int32))

    def one(params, c, key):
        carry, buf, st = c
        k_search, k_step, k_reset = jax.random.split(key, 3)
        s = carry.states
        game = carry.episode * games + lanes
        obs, mask = jax.vmap(env.observe)(s), jax.vmap(env.legal_mask)(s)
        action, weights, _ = policy(params, s, k_search, True)
        buf = replay.write(buf, obs, mask, weights, game)
        nxt = jax.vmap(env.step)(s, action, jax.random.split(k_step, games))
        out = jax.vmap(env.outcome)(nxt)
        bad = out.truncated | (out.error != 0)
        code = jnp.where(out.won, replay.WIN, jnp.where(bad, replay.NO_VALUE, replay.LOSS))
        buf = replay.finish(buf, game, out.done, code)
        st = Stats(st.decisions + games, st.games + out.done.sum(), st.wins + out.won.sum(),
                   st.truncated + (out.done & out.truncated).sum(), st.errors + (out.error != 0).sum(),
                   st.win_hp + jnp.where(out.won, out.hp, 0).sum())
        fresh = reset(k_reset)
        sel = lambda f, x: jnp.where(out.done.reshape((-1,) + (1,) * (x.ndim - 1)), f, x)
        return (Carry(jax.tree.map(sel, fresh, nxt), carry.episode + out.done), buf, st), None

    def run(params, carry, buf, key):
        z = jnp.int32(0)
        (carry, buf, st), _ = jax.lax.scan(lambda c, k: one(params, c, k), (carry, buf, Stats(z, z, z, z, z, z)),
                                           jax.random.split(key, chunk))
        return carry, buf, st

    return start, jax.jit(run, donate_argnums=(1, 2))


def make_eval(env: jaxsim.Env, policy, chunk: int):
    """play(params, bank, key) -> Outcome per start in `bank`, each played to the end without exploration
    (terminal states are absorbing, so finished lanes idle)."""
    def one(params, states, key):
        k_search, k_step = jax.random.split(key)
        n = jax.tree.leaves(states)[0].shape[0]
        action, _, _ = policy(params, states, k_search, False)
        return jax.vmap(env.step)(states, action, jax.random.split(k_step, n)), None

    @jax.jit
    def run(params, states, key):
        states, _ = jax.lax.scan(lambda s, k: one(params, s, k), states, jax.random.split(key, chunk))
        return states, jax.vmap(env.outcome)(states).done.all()

    def play(params, bank, key, max_chunks: int = 200):
        n = jax.tree.leaves(bank)[0].shape[0]
        k_init, key = jax.random.split(key)
        states = jax.vmap(env.init)(bank, jax.random.split(k_init, n))
        for _ in range(max_chunks):
            key, k = jax.random.split(key)
            states, all_done = run(params, states, k)
            if bool(all_done):
                break
        return jax.device_get(jax.vmap(env.outcome)(states))

    return play
