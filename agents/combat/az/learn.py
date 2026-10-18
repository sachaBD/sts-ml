"""Loss and jitted train step: policy cross-entropy to the search policy + value BCE to the game outcome."""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from agents.combat.az.search import NEG


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    lr: float
    weight_decay: float
    value_weight: float


def make_optimizer(cfg: TrainConfig):
    return optax.adamw(cfg.lr, weight_decay=cfg.weight_decay)


def make_train_step(net, optimizer, cfg: TrainConfig):
    def loss_fn(params, obs, mask, policy, value):
        logits, value_logit = jax.vmap(lambda o: net.apply(params, o))(obs)
        logp = jax.nn.log_softmax(jnp.where(mask, logits, NEG))
        policy_loss = -(policy * jnp.where(mask, logp, 0.0)).sum(-1).mean()
        value_loss = optax.sigmoid_binary_cross_entropy(value_logit, value).mean()
        return policy_loss + cfg.value_weight * value_loss, (policy_loss, value_loss)

    @jax.jit
    def step(params, opt_state, obs, mask, policy, value):
        (loss, (pl, vl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, obs, mask, policy, value)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, {"loss": loss, "policy_loss": pl, "value_loss": vl}

    return step
