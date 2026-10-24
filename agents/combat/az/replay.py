"""Replay buffer on the device: a ring of decisions plus a table of game outcomes. No host round trips.

Every decision is written as it happens, tagged with its game id. Value targets come from the outcome table,
written when the game ends, and are looked up at sampling time. A decision whose game hasn't ended yet still
trains the policy (its search target is known) but not the value.

Outcome codes: UNKNOWN (game running, or overwritten), LOSS, WIN, NO_VALUE (truncated or error: truncation is not
a loss, so it gives no value target).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

import jaxsim

UNKNOWN, LOSS, WIN, NO_VALUE = -1, 0, 1, 2


class Buffer(NamedTuple):
    obs: jaxsim.Obs        # leading [N]
    mask: jax.Array        # [N, A] bool
    policy: jax.Array      # [N, A] float32 search target
    game: jax.Array        # [N] int32 game id, -1 = empty
    pos: jax.Array         # [] int32 next write row
    size: jax.Array        # [] int32 rows filled
    outcome: jax.Array     # [M] int8 outcome code per game id slot
    outcome_game: jax.Array  # [M] int32 game id owning that slot (guards against wrap-around)


def empty(obs_example: jaxsim.Obs, A: int, capacity: int, games_table: int) -> Buffer:
    """obs_example: one observation (no batch dim). capacity must be a multiple of the self-play batch."""
    z = lambda x: jnp.zeros((capacity,) + x.shape, x.dtype)
    return Buffer(obs=jax.tree.map(z, obs_example), mask=jnp.zeros((capacity, A), bool),
                  policy=jnp.zeros((capacity, A), jnp.float32), game=jnp.full((capacity,), -1, jnp.int32),
                  pos=jnp.int32(0), size=jnp.int32(0),
                  outcome=jnp.full((games_table,), UNKNOWN, jnp.int8),
                  outcome_game=jnp.full((games_table,), -1, jnp.int32))


def write(buf: Buffer, obs, mask, policy, game) -> Buffer:
    """Append B decisions (leading [B]) at pos. capacity % B == 0, so a write never wraps mid-way."""
    B = mask.shape[0]
    put = lambda dst, src: jax.lax.dynamic_update_slice_in_dim(dst, src.astype(dst.dtype), buf.pos, 0)
    cap = buf.mask.shape[0]
    return buf._replace(obs=jax.tree.map(put, buf.obs, obs), mask=put(buf.mask, mask), policy=put(buf.policy, policy),
                        game=put(buf.game, game), pos=(buf.pos + B) % cap, size=jnp.minimum(buf.size + B, cap))


def finish(buf: Buffer, game, done, code) -> Buffer:
    """Record outcome `code` for the games (leading [B]) that ended (`done`)."""
    M = buf.outcome.shape[0]
    slot = jnp.where(done, game % M, M)                         # M: dropped
    return buf._replace(outcome=buf.outcome.at[slot].set(code.astype(jnp.int8), mode="drop"),
                        outcome_game=buf.outcome_game.at[slot].set(game, mode="drop"))


class Batch(NamedTuple):
    obs: jaxsim.Obs
    mask: jax.Array
    policy: jax.Array
    value: jax.Array         # [n] float32: 1 win, 0 loss
    value_weight: jax.Array  # [n] float32: 1 where the game ended in a win or loss
    game: jax.Array          # [n] int32 game id of each row


def sample(buf: Buffer, key, n: int) -> Batch:
    idx = jax.random.randint(key, (n,), 0, jnp.maximum(buf.size, 1))
    game = buf.game[idx]
    M = buf.outcome.shape[0]
    code = jnp.where(buf.outcome_game[game % M] == game, buf.outcome[game % M], UNKNOWN)
    return Batch(obs=jax.tree.map(lambda x: x[idx], buf.obs), mask=buf.mask[idx], policy=buf.policy[idx],
                 value=(code == WIN).astype(jnp.float32),
                 value_weight=((code == WIN) | (code == LOSS)).astype(jnp.float32), game=game)
