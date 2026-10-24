"""Learner: policy cross-entropy to the search policy (every decision) + value BCE to the game outcome (decisions
whose game ended in a win or loss), AdamW. K steps per jitted call, sampling from the device replay buffer."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from agents.combat.az import replay
from agents.combat.az.search import NEG


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    lr: float
    weight_decay: float
    value_weight: float


def make_optimizer(cfg: TrainConfig):
    return optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)


def make_train(net, optimizer, cfg: TrainConfig, steps: int):
    """train(params, opt_state, buffer, key) -> (params, opt_state, metrics averaged over `steps`)."""
    def loss_fn(params, b: replay.Batch):
        logits, value_logit = jax.vmap(lambda o, m: net.apply(params, o, m))(b.obs, b.mask)
        logp = jax.nn.log_softmax(jnp.where(b.mask, logits, NEG))
        policy_loss = -(b.policy * jnp.where(b.mask, logp, 0.0)).sum(-1).mean()
        vl = optax.sigmoid_binary_cross_entropy(value_logit, b.value)
        value_loss = (vl * b.value_weight).sum() / jnp.maximum(b.value_weight.sum(), 1.0)
        return policy_loss + cfg.value_weight * value_loss, dict(policy_loss=policy_loss, value_loss=value_loss,
                                                                 value_rows=b.value_weight.mean())

    def one(c, key):
        params, opt_state, buf = c
        b = replay.sample(buf, key, cfg.batch_size)
        (loss, m), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, buf), dict(loss=loss, **m)

    @jax.jit
    def train(params, opt_state, buf, key):
        (params, opt_state, _), m = jax.lax.scan(one, (params, opt_state, buf), jax.random.split(key, steps))
        return params, opt_state, jax.tree.map(jnp.mean, m)

    return train
