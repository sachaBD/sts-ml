"""Gumbel MuZero search (mctx) with the real simulator as the model, and root particles.

Particles: each game's root is searched P times with different keys (jaxsim's state holds no RNG, so a key is
a sampled future: draws and Champ rolls). The improved policies are averaged over particles.
Single-player, win-only: reward 1 on the transition into a win, discount 0 once the fight is over.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mctx

import jaxsim

NEG = -1e9


@dataclass(frozen=True)
class SearchConfig:
    simulations: int
    particles: int
    max_considered: int   # Gumbel top-m actions at the root
    gumbel_scale: float   # 1.0 explores (self-play), 0.0 is deterministic (evaluation)


def make_evaluate(env: jaxsim.Env, net):
    """(params, states[B]) -> masked logits [B, A], P(win) [B], legal mask [B, A]."""
    def evaluate(params, states):
        obs = jax.vmap(env.observe)(states)
        mask = jax.vmap(env.legal_mask)(states)
        logits, value_logit = jax.vmap(lambda o, m: net.apply(params, o, m))(obs, mask)
        return jnp.where(mask, logits, NEG), jax.nn.sigmoid(value_logit), mask
    return evaluate


def make_policy(env: jaxsim.Env, net, cfg: SearchConfig):
    evaluate = make_evaluate(env, net)

    def recurrent_fn(params, rng, action, states):
        keys = jax.random.split(rng, action.shape[0])
        was_done = jax.vmap(env.outcome)(states).done
        nxt = jax.vmap(env.step)(states, action, keys)
        out = jax.vmap(env.outcome)(nxt)
        logits, value, _ = evaluate(params, nxt)
        return mctx.RecurrentFnOutput(
            reward=(out.won & ~was_done).astype(jnp.float32),
            discount=(~out.done).astype(jnp.float32),
            prior_logits=logits,
            value=jnp.where(out.done, 0.0, value)), nxt

    def policy(params, states, key, explore: bool):
        """states [B] -> action [B], improved policy [B, A] (particle mean), search value [B].
        explore: sample the action from the improved policy (self-play); otherwise take its argmax."""
        key, k_act = jax.random.split(key)
        B, P = jax.tree.leaves(states)[0].shape[0], cfg.particles
        logits, value, mask = evaluate(params, states)
        rep = lambda x: jnp.repeat(x, P, axis=0)
        root = mctx.RootFnOutput(prior_logits=rep(logits), value=rep(value), embedding=jax.tree.map(rep, states))
        out = mctx.gumbel_muzero_policy(
            params, key, root, recurrent_fn, num_simulations=cfg.simulations, invalid_actions=rep(~mask),
            max_num_considered_actions=cfg.max_considered, gumbel_scale=cfg.gumbel_scale)
        weights = out.action_weights.reshape(B, P, -1).mean(1)
        weights = jnp.where(mask, weights, 0.0)
        root_value = out.search_tree.summary().value.reshape(B, P).mean(1)
        if explore:
            action = jax.random.categorical(k_act, jnp.where(mask, jnp.log(weights + 1e-12), NEG))
        else:
            action = jnp.argmax(weights, -1)
        return action.astype(jnp.int32), weights, root_value

    return policy
