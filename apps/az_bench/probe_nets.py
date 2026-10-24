"""Networks for the throughput probes (not agents): a no-compute floor and a small net shaped like the C++
value net v1 (~59k parameters)."""
import flax.linen as nn
import jax
import jax.numpy as jnp

import jaxsim

F32 = jnp.float32


class TrivialNet:
    """Uniform prior, P(win) = 0.5. No parameters, ~no compute: the floor of simulator + mctx."""
    def __init__(self, A):
        self.A = A

    def init(self, *_):
        return {}

    def apply(self, params, o, legal):
        return jnp.zeros((self.A,), F32), jnp.zeros((), F32)


class SmallNet(nn.Module):
    """Shaped like the C++ value net v1 (~63k): per-slot 2-layer MLP, monster MLP, sum+mean pooling, small head.
    Plus a per-slot policy logit (slot token + context) for the mctx prior."""
    spec: jaxsim.EnvSpec
    w: int = 64

    @nn.compact
    def __call__(self, o, legal):
        sp, w = self.spec, self.w
        f = lambda x: jnp.asarray(x, F32)
        clip = lambda x, hi: jnp.clip(jnp.asarray(x, jnp.int32), 0, hi - 1)
        D, P = sp.D, sp.P
        piles = f(o.piles)
        card = nn.Embed(sp.n_cards, 16)(clip(o.slot_card, sp.n_cards))
        x = jnp.concatenate([card, f(o.slot_upgraded)[:, None], f(o.slot_cost)[:, None], jnp.log1p(piles).T,
                             f(legal[:D])[:, None]], -1)
        tok = nn.relu(nn.Dense(w)(nn.relu(nn.Dense(w)(x))))                       # [D, w]
        live = (piles.sum(0) > 0)[:, None]
        n = jnp.maximum(live.sum(), 1)
        hand = piles[1][:, None] > 0
        pooled = jnp.concatenate([(tok * live).sum(0) / 10.0, (tok * live).sum(0) / n, (tok * hand).sum(0) / 5.0])
        enemy = nn.relu(nn.Dense(w)(jnp.concatenate([f(o.enemy[:3]) / 100.0, f(o.enemy_status) / 10.0,
                                                    jax.nn.one_hot(clip(o.enemy[3], sp.n_moves), sp.n_moves)])))
        player = nn.relu(nn.Dense(w)(jnp.concatenate([f(o.player) / 50.0, f(o.player_status) / 10.0,
                                                     f(o.turn)[None] / 20.0])))
        ctx = nn.relu(nn.Dense(w)(jnp.concatenate([pooled, enemy, player])))      # [w]
        slot_logit = nn.Dense(1)(nn.relu(nn.Dense(w)(tok + ctx)))[:, 0]
        pot = nn.Embed(sp.n_potions, 8)(clip(o.potions, sp.n_potions))
        pot_logit = nn.Dense(1)(jnp.concatenate([pot, jnp.broadcast_to(ctx, (P, w))], -1))[:, 0]
        ed = nn.Dense(2)(ctx)
        logits = jnp.concatenate([slot_logit, ed[:1], pot_logit, ed[1:]])
        return logits, nn.Dense(1)(ctx)[0]
