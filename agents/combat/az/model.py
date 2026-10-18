"""Policy/value network over jaxsim observations (integer inputs).

Topology (docs/research/topology/assessment.md: typed set tokens, shared encoders, sum pooling, a real head):
  card slot token  = MLP(emb(card) + emb(upgraded) + emb(cost))            context-free, one per slot
  pile pools       = count-weighted mean of slot tokens per pile, plus log1p(pile size)
  trunk            = LayerNorm -> Dense -> residual blocks over [pools, player, enemy, relics, turn]
  value            = Dense(1): logit of P(win)
  policy           = per-slot MLP([slot token, hand count, trunk]) for card actions;
                     Dense(trunk) for END_TURN, potions, DONE
Works on one observation; batch with jax.vmap.
"""
from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp

import jaxsim

F32 = jnp.float32
CONTRACT = "jaxsim_champ_v2"   # the observation layout this network reads (checked by test_az)


@dataclass(frozen=True)
class ModelConfig:
    width: int        # slot token width
    card_dim: int     # card id embedding size
    head_width: int   # trunk width
    head_blocks: int  # residual blocks in the trunk


class AZNet(nn.Module):
    spec: jaxsim.EnvSpec
    cfg: ModelConfig

    @nn.compact
    def __call__(self, o: jaxsim.Obs):
        sp, c = self.spec, self.cfg
        f = lambda x: x.astype(F32)
        clip = lambda x, lo, hi: jnp.clip(x.astype(jnp.int32), lo, hi)

        # card slot tokens
        tok = (nn.Embed(sp.n_cards, c.card_dim, name="card")(clip(o.slot_card, 0, sp.n_cards - 1))
               + nn.Embed(2, c.card_dim, name="upgraded")(clip(o.slot_upgraded, 0, 1))
               + nn.Embed(8, c.card_dim, name="cost")(clip(o.slot_cost + 2, 0, 7)))
        h = nn.relu(nn.Dense(c.width, name="slot_1")(tok))
        h = nn.relu(nn.Dense(c.width, name="slot_2")(h))                       # [D, W]
        counts = f(o.piles)                                                      # [5, D]
        size = counts.sum(-1, keepdims=True)                                     # [5, 1]
        pools = (counts @ h) / jnp.maximum(size, 1.0)                            # [5, W]

        # scalars (ints -> floats only here)
        player = f(o.player) / jnp.array([100.0, 100.0, 50.0, 5.0, 5.0])   # hp, max hp, block, energy, per turn
        enemy_num = f(o.enemy[:3]) / jnp.array([440.0, 440.0, 50.0])
        move = nn.Embed(sp.n_moves, 16, name="move")
        enemy = jnp.concatenate([enemy_num, move(clip(o.enemy[3], 0, sp.n_moves - 1)),
                                 move(clip(o.enemy[4], 0, sp.n_moves - 1)),
                                 nn.Embed(8, 8, name="ai_state")(clip(o.enemy[5], 0, 7))])
        status = jnp.concatenate([f(o.player_status), f(o.enemy_status)]) / 10.0
        relics = nn.Embed(sp.n_relics, 16, name="relic")(clip(o.relics, 0, sp.n_relics - 1))
        relics = (relics * (o.relics > 0)[:, None]).sum(0)
        turn = nn.Embed(64, 8, name="turn")(clip(o.turn, 0, 63))
        x = jnp.concatenate([pools.reshape(-1), jnp.log1p(size[:, 0]), player, enemy, status, relics, turn])

        # trunk
        x = nn.LayerNorm(name="trunk_in_norm")(x)
        t = nn.relu(nn.Dense(c.head_width, name="trunk_in")(x))
        for i in range(c.head_blocks):
            y = nn.LayerNorm(name=f"block_{i}_norm")(t)
            y = nn.Dense(c.head_width, name=f"block_{i}_fc2")(nn.relu(nn.Dense(c.head_width, name=f"block_{i}_fc1")(y)))
            t = t + y
        t = nn.LayerNorm(name="trunk_out_norm")(t)

        value_logit = nn.Dense(1, name="value")(t)[0]
        hand = nn.Embed(11, 8, name="hand_count")(clip(o.piles[jaxsim.PILE_HAND], 0, 10))
        ctx = jnp.broadcast_to(nn.Dense(c.width, name="policy_ctx")(t), (sp.D, c.width))
        p = nn.relu(nn.Dense(c.width, name="policy_1")(jnp.concatenate([h, hand, ctx], -1)))
        slot_logits = nn.Dense(1, name="policy_2")(p)[:, 0]                    # [D]
        other = nn.Dense(sp.A - sp.D, name="policy_other")(t)                  # END_TURN, potions, DONE
        return jnp.concatenate([slot_logits, other]), value_logit
