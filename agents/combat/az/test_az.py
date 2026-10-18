"""AZ agent: network shapes, search legality, self-play records, replay targets, and that learning reduces loss."""
import unittest

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jaxsim
except ImportError as e:  # JAX stack is optional (requirements-jax.txt)
    raise unittest.SkipTest(f"JAX stack not installed: {e}")

from agents.combat.az import decks, learn, model, replay, search, selfplay

SPEC = jaxsim.DEFAULT_SPEC
MODEL = model.ModelConfig(width=16, card_dim=8, head_width=32, head_blocks=1)
SEARCH = search.SearchConfig(simulations=8, particles=2, max_considered=4, gumbel_scale=1.0)
DECKS = decks.DeckConfig(count=8, extra_min=0, extra_max=5, upgrade_prob=0.5, hp_min=60, hp_max=300, seed=0)


class ContractTests(unittest.TestCase):
    def test_jaxsim_contract_matches_the_network(self):
        self.assertEqual(jaxsim.CONTRACT, model.CONTRACT, "jaxsim observation layout changed: update AZNet")


class AZTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = jaxsim.make_env(SPEC)
        cls.net = model.AZNet(SPEC, MODEL)
        cls.bank = decks.make_bank(DECKS, SPEC)
        s0 = cls.env.init(jax.tree.map(lambda x: x[0], cls.bank), jax.random.key(0))
        cls.params = cls.net.init(jax.random.key(1), cls.env.observe(s0))

    def test_network_shapes(self):
        s = self.env.init(jax.tree.map(lambda x: x[0], self.bank), jax.random.key(0))
        logits, v = self.net.apply(self.params, self.env.observe(s))
        self.assertEqual(logits.shape, (SPEC.A,))
        self.assertEqual(v.shape, ())

    def test_search_picks_legal_actions(self):
        policy = jax.jit(search.make_policy(self.env, self.net, SEARCH), static_argnums=3)
        states = jax.vmap(self.env.init)(self.bank, jax.random.split(jax.random.key(2), DECKS.count))
        mask = jax.vmap(self.env.legal_mask)(states)
        for explore in (True, False):
            action, weights, value = policy(self.params, states, jax.random.key(3), explore)
            self.assertTrue(bool(mask[jnp.arange(DECKS.count), action].all()))
            np.testing.assert_allclose(np.asarray(weights.sum(-1)), 1.0, rtol=1e-5)
            self.assertFalse(bool((weights * ~mask).any()))
            self.assertTrue(bool(((value >= 0) & (value <= 1)).all()))

    def test_selfplay_to_learning(self):
        policy = search.make_policy(self.env, self.net, SEARCH)
        start, run = selfplay.make_selfplay(self.env, policy, self.bank, games=4, chunk=64, explore=True)
        carry = start(jax.random.key(4))
        buf = replay.Replay(capacity=4096, seed=0)
        for i in range(6):
            carry, rec = run(self.params, carry, jax.random.key(10 + i))
            self.assertEqual(rec.mask.shape, (64, 4, SPEC.A))
            self.assertEqual(int(np.asarray(rec.error).max()), 0)
            buf.add(rec)
        self.assertGreater(buf.games, 0)
        self.assertGreater(buf.size, 0)
        self.assertTrue(set(np.unique(buf.data["value"][: buf.size])) <= {0.0, 1.0})

        cfg = learn.TrainConfig(batch_size=64, lr=3e-3, weight_decay=0.0, value_weight=1.0)
        opt = learn.make_optimizer(cfg)
        step = learn.make_train_step(self.net, opt, cfg)
        params, opt_state = self.params, opt.init(self.params)
        batch = buf.sample(cfg.batch_size)  # fixed batch: loss must fall when fitting it
        args = (replay.batch_obs(batch), batch["mask"], batch["policy"], batch["value"])
        first = None
        for _ in range(30):
            params, opt_state, m = step(params, opt_state, *args)
            first = float(m["loss"]) if first is None else first
        self.assertLess(float(m["loss"]), first)


if __name__ == "__main__":
    unittest.main()
