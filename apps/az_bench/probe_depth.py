"""Per-decision search time and tree depth over the course of fights, toy fight or jaxsim. Same mctx search as
agents/combat/az/search.py (including the uniform prior below finished fights).

  PYTHONPATH=. [NET=small] [MAXD=16] [DUMP=1] .venv/bin/python -u -m apps.az_bench.probe_depth toy|jaxsim B DECISIONS
NET=small: the C++-like probe net (default: trivial, uniform prior over legal moves, value 0.5).
MAXD: mctx max_depth. DUMP=1: print the deepest path once a tree is >= 50 deep.
Self-play loop: search -> sample action from improved policy -> step -> finished lanes restart.
The toy fight (ToyEnv) shows mctx + this stack alone are fast: ~4-5 ms/simulation step at 512 games over 30 moves.
"""
import os
import sys
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mctx
import numpy as np

A, NEG = 55, -1e9
I32 = jnp.int32


class Outcome(NamedTuple):
    done: jax.Array
    won: jax.Array
    truncated: jax.Array
    error: jax.Array
    hp: jax.Array


class ToyState(NamedTuple):
    php: jax.Array
    ehp: jax.Array
    block: jax.Array
    energy: jax.Array
    hand: jax.Array      # [5] card 0 strike (1 energy, 6 dmg), 1 defend (1, 5 block), 2 bash (2, 10 dmg), 3 empty
    turn: jax.Array
    done: jax.Array
    won: jax.Array
    pad: jax.Array       # [300] int32: makes the state ~1.2 KB like jaxsim's


COST = jnp.array([1, 1, 2, 9])


class ToyEnv:
    """Absorbing terminals, all-False legal mask once done: like jaxsim."""
    def init(self, key):
        return ToyState(I32(80), I32(120), I32(0), I32(3), jax.random.randint(key, (5,), 0, 3), I32(0),
                        jnp.bool_(False), jnp.bool_(False), jnp.zeros((300,), I32))

    def legal_mask(self, s):
        slots = (s.hand != 3) & (COST[s.hand] <= s.energy)
        m = jnp.zeros((A,), bool).at[:5].set(slots).at[48].set(True)
        return m & ~s.done

    def step(self, s, a, key):
        i = jnp.clip(a, 0, 4)
        c = s.hand[i]
        play = a < 5
        dmg = jnp.where(play, jnp.array([6, 0, 10, 0])[c], 0)
        blk = jnp.where(play, jnp.array([0, 5, 0, 0])[c], 0)
        k1, k2 = jax.random.split(key)
        hit = jax.random.randint(k1, (), 8, 21)
        end = a == 48
        ehp = s.ehp - dmg
        block = s.block + blk
        php = jnp.where(end, s.php - jnp.maximum(hit - block, 0), s.php)
        hand = jnp.where(end, jax.random.randint(k2, (5,), 0, 3), s.hand.at[i].set(jnp.where(play, 3, c)))
        n = ToyState(php, ehp, jnp.where(end, 0, block), jnp.where(end, 3, s.energy - jnp.where(play, COST[c], 0)),
                     hand, s.turn + end, (ehp <= 0) | (php <= 0) | (s.turn + end >= 50), (ehp <= 0) & (php > 0), s.pad + 1)
        return jax.tree.map(lambda x, y: jnp.where(s.done, x, y), s, n)

    def outcome(self, s):
        return Outcome(s.done, s.won, s.done & ~s.won & (s.php > 0), I32(0), s.php)

    def observe(self, s):
        return s


def make_jaxsim():
    import jaxsim
    from agents.combat.az import decks, selfplay
    from apps.az_bench.bench import REFERENCE_DECKS
    spec = jaxsim.DEFAULT_SPEC
    env = jaxsim.make_env(spec)
    bank = decks.random_bank(REFERENCE_DECKS, spec)
    init = lambda key: env.init(jax.tree.map(lambda x: x[0], selfplay.sample_starts(bank, key, 1)), key)
    return env, init


def main():
    which, B, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    S = 128
    if which == "toy":
        env = ToyEnv()
        init = env.init
    else:
        env, init = make_jaxsim()
    v = lambda f: jax.vmap(f)

    if os.environ.get("NET") == "small":
        from apps.az_bench.probe_nets import SmallNet
        import jaxsim as _js
        net = SmallNet(_js.DEFAULT_SPEC)
        s0 = init(jax.random.key(0))
        params = net.init(jax.random.key(0), env.observe(s0), env.legal_mask(s0))
    else:
        net = None

    def evaluate(states):
        mask = v(env.legal_mask)(states)
        if net is None:
            _ = v(env.observe)(states)
            return jnp.where(mask, 0.0, NEG), jnp.full(mask.shape[:1], 0.5), mask
        logits, vl = jax.vmap(lambda o, m: net.apply(params, o, m))(v(env.observe)(states), mask)
        return jnp.where(mask, logits, NEG), jax.nn.sigmoid(vl), mask

    def recurrent_fn(params, rng, action, states):     # as agents/combat/az/search.py
        was_done = v(env.outcome)(states).done
        nxt = v(env.step)(states, action, jax.random.split(rng, action.shape[0]))
        out = v(env.outcome)(nxt)
        logits, value, _ = evaluate(nxt)
        logits = jnp.where(out.done[:, None], 0.0, logits)   # the search.py fix
        return mctx.RecurrentFnOutput(reward=(out.won & ~was_done).astype(jnp.float32),
                                      discount=(~out.done).astype(jnp.float32), prior_logits=logits,
                                      value=jnp.where(out.done, 0.0, value)), nxt

    @jax.jit
    def search(states, key):
        logits, value, mask = evaluate(states)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=states)
        out = mctx.gumbel_muzero_policy({}, key, root, recurrent_fn, num_simulations=S, invalid_actions=~mask,
                                        max_num_considered_actions=16, gumbel_scale=1.0,
                                        max_depth=int(os.environ['MAXD']) if os.environ.get('MAXD') else None)
        w = jnp.where(mask, out.action_weights, 0.0)
        return w, out.search_tree.parents, out.search_tree.embeddings, out.search_tree.action_from_parent

    @jax.jit
    def advance(states, w, key):
        k1, k2, k3 = jax.random.split(key, 3)
        a = jax.random.categorical(k1, jnp.where(w > 0, jnp.log(w + 1e-12), NEG))
        nxt = v(env.step)(states, a, jax.random.split(k2, B))
        done = v(env.outcome)(nxt).done
        fresh = v(init)(jax.random.split(k3, B))
        return jax.tree.map(lambda f, x: jnp.where(done.reshape((-1,) + (1,) * (x.ndim - 1)), f, x), fresh, nxt), done.sum()

    def depth_stats(parents, emb):
        par = np.asarray(parents)                     # [B, N], -1 for root / unused
        n = par.shape[1]
        d = np.zeros_like(par)
        for j in range(1, n):
            p = par[:, j]
            d[:, j] = np.where(p >= 0, d[np.arange(len(p)), np.maximum(p, 0)] + 1, 0)
        done = np.asarray(jax.vmap(jax.vmap(env.outcome))(emb).done)   # [B, N]
        return d.max(1), (done & (par >= 0)).sum(1)

    dumped = False

    def dump(parents, emb, afp, lane):
        par = np.asarray(parents)[lane]
        d = np.zeros(len(par), int)
        for j in range(1, len(par)):
            d[j] = d[par[j]] + 1 if par[j] >= 0 else 0
        node = int(d.argmax())
        path = []
        while node >= 0:
            path.append(node)
            node = int(par[node])
        path = path[::-1]
        e = jax.tree.map(lambda x: x[lane], emb)
        print(f"--- deepest chain, lane {lane}, depth {len(path) - 1}", flush=True)
        for depth, n in enumerate(path):
            s1 = jax.tree.map(lambda x: x[n], e)
            o = env.observe(s1)
            m = np.asarray(env.legal_mask(s1))
            out = env.outcome(s1)
            if depth < 12 or depth % 10 == 0 or depth > len(path) - 4:
                print(f"  depth {depth:3d} node {n:3d} action_in {int(np.asarray(afp)[lane, n]):3d} legal {np.flatnonzero(m).tolist()} "
                      f"turn {int(o.turn)} php {int(o.player[0])} block {int(o.player[2])} energy {int(o.player[3])} "
                      f"ehp {int(o.enemy[0])} done {bool(out.done)} won {bool(out.won)} trunc {bool(out.truncated)} err {int(out.error)}", flush=True)

    states = v(init)(jax.random.split(jax.random.key(1), B))
    t0 = time.perf_counter()
    for t in range(T):
        tc = time.perf_counter()
        w, parents, emb, afp = jax.block_until_ready(search(states, jax.random.key(100 + t)))
        dt = time.perf_counter() - tc
        maxd, done_nodes = depth_stats(parents, emb)
        if os.environ.get('DUMP') and maxd.max() >= 50 and not dumped:
            dumped = True
            dump(parents, emb, afp, int(maxd.argmax()))
        states, finished = advance(states, w, jax.random.key(200 + t))
        print(f"[{time.perf_counter() - t0:6.1f}s] {which} B={B} decision {t:3d}: {dt * 1000:8.1f} ms  "
              f"{dt / S * 1000:6.2f} ms/sim-step  tree depth max {maxd.max():3d} mean {maxd.mean():5.1f}  "
              f"terminal nodes/tree {done_nodes.mean():5.1f}  fights finished {int(finished)}", flush=True)


if __name__ == "__main__":
    main()
