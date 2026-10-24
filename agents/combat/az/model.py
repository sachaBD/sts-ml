"""Policy/value network over jaxsim observations: a small transformer over card-slot, potion and global tokens.

Tokens (one observation; batch with jax.vmap):
  slot d (D of them)   emb(card) + emb(upgrades) + emb(cost) + Dense(log1p pile counts, misc, known-top, legal)
  potion slot (P)      emb(potion) + Dense(count, legal)
  player (1)           Dense(hp, block, energy, statuses, turn, choice state) + emb(choice kind) + pooled relics
  enemy (1)            Dense(hp, block, statuses) + emb(intent) + emb(previous move) + emb(AI state)
Empty slots and potions are masked out as attention keys. Pre-LayerNorm transformer blocks.
  policy  slot logits from slot tokens, potion logits from potion tokens, END_TURN / DONE from the player token
  value   logit of P(win) from [player, enemy, mean of live tokens]
Every slot of every pile is a token (not only the hand): choices pick from discard / exhaust slots, and the
draw pile's content matters for planning. The legal mask is an input.
"""
from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp

import jaxsim

F32 = jnp.float32
CONTRACT = "jaxsim_champ_v1.0"   # the observation layout this network reads (checked by test_az)


@dataclass(frozen=True)
class ModelConfig:
    d: int            # token width
    layers: int       # transformer blocks
    heads: int        # attention heads
    mlp_ratio: int    # block MLP width = d * mlp_ratio


class Block(nn.Module):
    d: int
    heads: int
    mlp_ratio: int

    @nn.compact
    def __call__(self, x, mask):
        h = nn.LayerNorm()(x)
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.heads, qkv_features=self.d, out_features=self.d)(
            h, mask=mask)
        h = nn.LayerNorm()(x)
        return x + nn.Dense(self.d)(nn.gelu(nn.Dense(self.d * self.mlp_ratio)(h)))


class AZNet(nn.Module):
    spec: jaxsim.EnvSpec
    cfg: ModelConfig

    @nn.compact
    def __call__(self, o: jaxsim.Obs, legal):
        sp, d = self.spec, self.cfg.d
        f = lambda x: jnp.asarray(x, F32)
        clip = lambda x, hi: jnp.clip(jnp.asarray(x, jnp.int32), 0, hi - 1)
        D, P = sp.D, sp.P

        # card slots
        piles = f(o.piles)                                                     # [5, D]
        idx = jnp.arange(D)
        slot_num = jnp.concatenate([
            jnp.log1p(piles).T,                                               # [D, 5]
            (f(o.slot_misc) / 10.0)[:, None],
            (o.known_top[0] == idx)[:, None].astype(F32), (o.known_top[1] == idx)[:, None].astype(F32),
            f(legal[:D])[:, None]], -1)
        slots = (nn.Embed(sp.n_cards, d, name="card")(clip(o.slot_card, sp.n_cards))
                 + nn.Embed(16, d, name="upgrades")(clip(o.slot_upgraded, 16))
                 + nn.Embed(12, d, name="cost")(clip(o.slot_cost + 3, 12))
                 + nn.Dense(d, name="slot_num")(slot_num))
        slot_live = piles.sum(0) > 0

        # potions
        pot_legal = f(legal[sp.END_TURN + 1: sp.END_TURN + 1 + P])
        pots = (nn.Embed(sp.n_potions, d, name="potion")(clip(o.potions, sp.n_potions))
                + nn.Dense(d, name="potion_num")(jnp.stack([f(o.potion_counts), pot_legal], -1)))
        pot_live = o.potions > 0

        # player (with relics and the pending choice)
        relics = nn.Embed(sp.n_relics, d, name="relic")(clip(o.relics, sp.n_relics))
        relics = relics + nn.Dense(d, name="relic_ctr")(jnp.sign(f(o.relic_counters)[:, None])
                                                         * jnp.log1p(jnp.abs(f(o.relic_counters)))[:, None])
        relics = (relics * (o.relics > 0)[:, None]).sum(0)
        p = f(o.player) / jnp.array([100.0, 100.0, 50.0, 5.0, 5.0])
        player_num = jnp.concatenate([p, f(o.player_status) / 10.0, f(o.turn)[None] / 50.0,
                                      f(o.potion_capacity)[None] / 5.0, f(o.choice[1:]) / 10.0])
        player = (nn.Dense(d, name="player_num")(player_num) + relics
                  + nn.Embed(16, d, name="choice_kind")(clip(o.choice[0], 16)))

        # enemy
        e = f(o.enemy[:3]) / jnp.array([440.0, 440.0, 50.0])
        move = nn.Embed(sp.n_moves, d, name="move")
        enemy = (nn.Dense(d, name="enemy_num")(jnp.concatenate([e, f(o.enemy_status) / 10.0]))
                 + move(clip(o.enemy[3], sp.n_moves)) + nn.Dense(d, name="last_move")(move(clip(o.enemy[4], sp.n_moves)))
                 + nn.Embed(8, d, name="ai_state")(clip(o.enemy[5], 8)))

        x = jnp.concatenate([slots, pots, player[None], enemy[None]])          # [D + P + 2, d]
        live = jnp.concatenate([slot_live, pot_live, jnp.array([True, True])])
        mask = jnp.broadcast_to(live[None, None, :], (1, x.shape[0], x.shape[0]))
        for i in range(self.cfg.layers):
            x = Block(d, self.cfg.heads, self.cfg.mlp_ratio, name=f"block_{i}")(x, mask)
        x = nn.LayerNorm(name="out_norm")(x)

        slot_logits = nn.Dense(1, name="policy_slot")(x[:D])[:, 0]
        pot_logits = nn.Dense(1, name="policy_potion")(x[D:D + P])[:, 0]
        end_done = nn.Dense(2, name="policy_end_done")(x[D + P])
        logits = jnp.concatenate([slot_logits, end_done[:1], pot_logits, end_done[1:]])   # jaxsim action layout

        pooled = (x * live[:, None]).sum(0) / jnp.maximum(live.sum(), 1)
        v = nn.gelu(nn.Dense(d, name="value_hidden")(jnp.concatenate([x[D + P], x[D + P + 1], pooled])))
        return logits, nn.Dense(1, name="value")(v)[0]
