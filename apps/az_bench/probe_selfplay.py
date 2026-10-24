"""Self-play throughput probe: the real az search + self-play path with a chosen network.

  PYTHONPATH=. .venv/bin/python -u -m apps.az_bench.probe_selfplay --net trivial|small|big --games 512 2048 --sims 128
Times `repeat` calls of `chunk` decisions after one warm-up call (compile + first). Measured mid-fight (moves ~5-16).
"""
import argparse
import time

import jax

import jaxsim
from agents.combat.az import decks, model, replay, search, selfplay
from apps.az_bench.bench import REFERENCE_DECKS, REFERENCE_MODEL
from apps.az_bench.probe_nets import SmallNet, TrivialNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", choices=["trivial", "small", "big"], required=True)
    p.add_argument("--games", type=int, nargs="+", default=[256, 2048])
    p.add_argument("--sims", type=int, default=128)
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--repeat", type=int, default=2)
    a = p.parse_args()
    t0 = time.perf_counter()
    lap = lambda: f"[{time.perf_counter() - t0:5.1f}s]"

    spec = jaxsim.DEFAULT_SPEC
    env = jaxsim.make_env(spec)
    bank = decks.random_bank(REFERENCE_DECKS, spec)
    net = {"trivial": lambda: TrivialNet(spec.A), "small": lambda: SmallNet(spec),
           "big": lambda: model.AZNet(spec, REFERENCE_MODEL)}[a.net]()
    s0 = env.init(jax.tree.map(lambda x: x[0], bank), jax.random.key(0))
    params = net.init(jax.random.key(0), env.observe(s0), env.legal_mask(s0))
    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"{lap()} {jax.devices()[0]} net={a.net} params={n_params} sims={a.sims}", flush=True)
    cfg = search.SearchConfig(simulations=a.sims, particles=1, max_considered=16, gumbel_scale=1.0)
    for g in a.games:
        start, run = selfplay.make_selfplay(env, search.make_policy(env, net, cfg), bank, g, a.chunk)
        carry = start(jax.random.key(4))
        buf = replay.empty(env.observe(s0), spec.A, g * a.chunk * (a.repeat + 1), 1 << 16)
        tc = time.perf_counter()
        carry, buf, _ = jax.block_until_ready(run(params, carry, buf, jax.random.key(5)))
        compile_s = time.perf_counter() - tc
        tr = time.perf_counter()
        fights = errors = 0
        for i in range(a.repeat):
            carry, buf, st = jax.block_until_ready(run(params, carry, buf, jax.random.key(6 + i)))
            fights += int(st.games)
            errors += int(st.errors)
        dt = time.perf_counter() - tr
        dec = g * a.chunk * a.repeat
        print(f"{lap()} net={a.net} games={g:5d} sims={a.sims}: {dec / dt:10,.0f} decisions/s  "
              f"{dec * a.sims / dt:12,.0f} simulations/s  {dt / (a.chunk * a.repeat) * 1000:8.1f} ms/decision-step  "
              f"fights_done={fights} errors={errors}  (compile+first {compile_s:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
