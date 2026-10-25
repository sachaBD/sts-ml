"""AZ agent: network shapes, search legality, device replay targets, self-play into the buffer, deck files, and
that learning reduces loss."""
import json
import tempfile
import unittest
from pathlib import Path

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import jaxsim
except ImportError as e:  # JAX stack missing from this environment
    raise unittest.SkipTest(f"JAX stack not installed: {e}")

from agents.combat.az import decks, learn, model, replay, search, selfplay

SPEC = jaxsim.DEFAULT_SPEC
MODEL = model.ModelConfig(d=16, layers=1, heads=2, mlp_ratio=2)
SEARCH = search.SearchConfig(simulations=4, particles=2, max_considered=4, gumbel_scale=1.0)
DECKS = decks.RandomDecks(count=8, extra_min=0, extra_max=5, upgrade_prob=0.5, hp_min=60, hp_max=300, seed=0)


class ContractTests(unittest.TestCase):
    def test_jaxsim_contract_matches_the_network(self):
        self.assertEqual(jaxsim.CONTRACT, model.CONTRACT, "jaxsim observation layout changed: update AZNet")


class DeckFileTests(unittest.TestCase):
    def test_load_accepts_and_reports_rejections(self):
        data = {"decks": [
            {"name": "ok", "hp": 50, "max_hp": 80, "cards": ["BASH+", "STRIKE_RED", ["SEARING_BLOW", 2]],
             "relics": ["VAJRA", ["PEN_NIB", 4]], "potions": ["FIRE_POTION"]},
            {"name": "bad card", "hp": 50, "cards": ["HAVOC"]},
            {"name": "bad hp", "hp": 90, "max_hp": 80, "cards": ["BASH"]}]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "decks.json"
            p.write_text(json.dumps(data))
            loaded = decks.load_file(p, SPEC)
        self.assertEqual(loaded.names, ["ok"])
        self.assertEqual([n for n, _ in loaded.rejected], ["bad card", "bad hp"])
        self.assertEqual(int(loaded.bank.count.sum()), 3)


class ReplayTests(unittest.TestCase):
    def test_value_targets_come_from_finished_games_only(self):
        env = jaxsim.make_env(SPEC)
        s = env.init(jaxsim.make_start(decks.STARTER, hp=80, spec=SPEC), jax.random.key(0))
        obs, mask = env.observe(s), env.legal_mask(s)
        buf = replay.empty(obs, SPEC.A, capacity=8, games_table=4)
        rows = lambda x: jax.tree.map(lambda a: jnp.stack([a] * 4), x)
        buf = replay.write(buf, rows(obs), rows(mask), jnp.ones((4, SPEC.A)) / SPEC.A, jnp.arange(4, dtype=jnp.int32))
        done = jnp.array([True, True, True, False])
        code = jnp.array([replay.WIN, replay.LOSS, replay.NO_VALUE, replay.WIN])
        buf = replay.finish(buf, jnp.arange(4, dtype=jnp.int32), done, code)
        b = replay.sample(buf, jax.random.key(1), 64)
        game = np.asarray(buf.game)[:4]
        self.assertEqual(game.tolist(), [0, 1, 2, 3])
        v, w = np.asarray(b.value), np.asarray(b.value_weight)
        self.assertTrue(set(zip(v.tolist(), w.tolist())) <= {(1.0, 1.0), (0.0, 1.0), (0.0, 0.0)})
        # games 4..7 reuse outcome slots 0..3 (table of 4): games 0..3 lose their value targets, 4..7 get theirs
        buf = replay.write(buf, rows(obs), rows(mask), jnp.ones((4, SPEC.A)) / SPEC.A, jnp.arange(4, 8, dtype=jnp.int32))
        buf = replay.finish(buf, jnp.arange(4, 8, dtype=jnp.int32), jnp.array([True] * 4), code)
        b = replay.sample(buf, jax.random.key(2), 256)
        g, w = np.asarray(b.game), np.asarray(b.value_weight)
        self.assertEqual(float(w[g < 4].sum()), 0.0)
        self.assertEqual(sorted(set(g[w > 0].tolist())), [4, 5, 7])   # 6 was NO_VALUE


class AZTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = jaxsim.make_env(SPEC)
        cls.net = model.AZNet(SPEC, MODEL)
        cls.bank = decks.random_bank(DECKS, SPEC)
        s0 = cls.env.init(jax.tree.map(lambda x: x[0], cls.bank), jax.random.key(0))
        cls.obs0 = cls.env.observe(s0)
        cls.params = cls.net.init(jax.random.key(1), cls.obs0, cls.env.legal_mask(s0))

    def test_network_shapes(self):
        s = self.env.init(jax.tree.map(lambda x: x[0], self.bank), jax.random.key(0))
        logits, v = self.net.apply(self.params, self.env.observe(s), self.env.legal_mask(s))
        self.assertEqual(logits.shape, (SPEC.A,))
        self.assertEqual(v.shape, ())
        self.assertTrue(bool(jnp.isfinite(logits).all()))

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
        games, chunk = 4, 32
        policy = search.make_policy(self.env, self.net, SEARCH)
        start, run = selfplay.make_selfplay(self.env, policy, self.bank, games=games, chunk=chunk)
        carry = start(jax.random.key(4))
        buf = replay.empty(self.obs0, SPEC.A, capacity=4096, games_table=1024)
        total = None
        for i in range(4):
            carry, buf, st = run(self.params, carry, buf, jax.random.key(10 + i))
            st = jax.device_get(st)
            total = st if total is None else jax.tree.map(lambda a, b: a + b, total, st)
        self.assertEqual(int(total.decisions), 4 * games * chunk)
        self.assertEqual(int(buf.size), 4 * games * chunk)
        self.assertEqual(int(total.errors), 0)
        self.assertGreater(int(total.games), 0)

        cfg = learn.TrainConfig(batch_size=64, lr=3e-3, weight_decay=0.0, value_weight=1.0)
        opt = learn.make_optimizer(cfg)
        train = learn.make_train(self.net, opt, cfg, steps=30)
        params, opt_state = self.params, opt.init(self.params)
        _, _, m0 = learn.make_train(self.net, opt, cfg, steps=1)(params, opt_state, buf, jax.random.key(5))
        params, opt_state, m = train(params, opt_state, buf, jax.random.key(6))
        params, opt_state, m = train(params, opt_state, buf, jax.random.key(7))
        self.assertLess(float(m["policy_loss"]), float(m0["policy_loss"]))


if __name__ == "__main__":
    unittest.main()
