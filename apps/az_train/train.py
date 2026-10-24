#!/usr/bin/env python3
"""AlphaZero-style self-play training on the JAX simulator: Ironclad vs The Champ, A20 (agents/combat/az).

Self-play, the replay buffer and the learner all run on the device; the host only orchestrates, logs and saves.

Every setting is explicit in the TOML (no defaults):
  [run]       id
  [decks]     train, eval: deck files (agents/combat/az/decks.py format; paths relative to the repo root)
  [model]     agents.combat.az.model.ModelConfig: d, layers, heads, mlp_ratio
  [search]    agents.combat.az.search.SearchConfig for self-play
  [selfplay]  games (parallel lanes), chunk (decisions per jitted call)
  [train]     iterations, chunks_per_iteration, steps_per_iteration, batch_size, lr, weight_decay, value_weight,
              replay_capacity (a multiple of games), games_table, min_replay, seed
  [eval]      every (iterations), simulations, particles, repeats (games per eval deck), chunk
Writes to out/: config.json, decks_report.txt, metrics.jsonl (one line per iteration),
checkpoints/state-<iteration>.msgpack (params + optimizer), params.msgpack (final), summary.json.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any

from apps.common.app import check_keys, main, required

log = logging.getLogger(__name__)

TABLES = {"run", "decks", "model", "search", "selfplay", "train", "eval"}
DECKS = {"train": str, "eval": str}
SELFPLAY = {"games": int, "chunk": int}
TRAIN = {"iterations": int, "chunks_per_iteration": int, "steps_per_iteration": int, "batch_size": int, "lr": float,
         "weight_decay": float, "value_weight": float, "replay_capacity": int, "games_table": int, "min_replay": int,
         "seed": int}
EVAL = {"every": int, "simulations": int, "particles": int, "repeats": int, "chunk": int}


def typed(table: dict, spec: dict, name: str) -> dict:
    check_keys(table, spec, name)
    return {k: required(table, k, name, kind) for k, kind in spec.items()}


def dataclass_from(cls, table: dict, name: str):
    kinds = {"int": int, "float": float}
    fields = {f.name: kinds.get(f.type, f.type) if isinstance(f.type, str) else f.type for f in dataclasses.fields(cls)}
    return cls(**typed(table, fields, name))


def parse(config: dict[str, Any]):
    from agents.combat.az import model, search
    check_keys(config, TABLES, "config")
    for t in TABLES:
        required(config, t, "config", dict)
    c = dict(decks=typed(config["decks"], DECKS, "decks"),
             model=dataclass_from(model.ModelConfig, config["model"], "model"),
             search=dataclass_from(search.SearchConfig, config["search"], "search"),
             selfplay=typed(config["selfplay"], SELFPLAY, "selfplay"),
             train=typed(config["train"], TRAIN, "train"),
             eval=typed(config["eval"], EVAL, "eval"))
    if c["train"]["replay_capacity"] % c["selfplay"]["games"]:
        raise ValueError("train.replay_capacity must be a multiple of selfplay.games")
    return c


def load_decks(path: str, spec, out: Path, what: str):
    from agents.combat.az import decks
    loaded = decks.load_file(path, spec)
    report = f"{what} decks ({path}): {loaded.report()}"
    log.info("%s", report)
    with (out / "decks_report.txt").open("a") as f:
        f.write(report + "\n" + "".join(f"  rejected {n}: {r}\n" for n, r in loaded.rejected))
    if loaded.bank is None:
        raise ValueError(f"no playable {what} decks in {path}")
    return loaded


def run(config: dict[str, Any], config_path: Path, out: Path) -> int:
    import jax
    import numpy as np
    from flax import serialization

    import jaxsim
    from agents.combat.az import learn, model, replay, search, selfplay

    c = parse(config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "decks_report.txt").write_text("")
    log.info("devices %s, jaxsim %s", jax.devices(), jaxsim.CONTRACT)

    spec = jaxsim.DEFAULT_SPEC
    env = jaxsim.make_env(spec)
    net = model.AZNet(spec, c["model"])
    train_decks = load_decks(c["decks"]["train"], spec, out, "train")
    eval_decks = load_decks(c["decks"]["eval"], spec, out, "eval")

    tr, sp, ev = c["train"], c["selfplay"], c["eval"]
    key = jax.random.key(tr["seed"])
    key, k_init = jax.random.split(key)
    s0 = env.init(jax.tree.map(lambda x: x[0], train_decks.bank), k_init)
    obs0 = env.observe(s0)
    params = net.init(k_init, obs0, env.legal_mask(s0))
    n_params = sum(x.size for x in jax.tree.leaves(params))
    log.info("parameters %d", n_params)

    tcfg = learn.TrainConfig(tr["batch_size"], tr["lr"], tr["weight_decay"], tr["value_weight"])
    opt = learn.make_optimizer(tcfg)
    opt_state = opt.init(params)
    train = learn.make_train(net, opt, tcfg, tr["steps_per_iteration"])
    policy = search.make_policy(env, net, c["search"])
    start, play = selfplay.make_selfplay(env, policy, train_decks.bank, sp["games"], sp["chunk"])
    key, k = jax.random.split(key)
    carry = start(k)
    buf = replay.empty(obs0, spec.A, tr["replay_capacity"], tr["games_table"])

    eval_policy = search.make_policy(env, net, search.SearchConfig(ev["simulations"], ev["particles"],
                                                                    c["search"].max_considered, 0.0))
    evaluate = selfplay.make_eval(env, eval_policy, ev["chunk"])
    n_eval = len(eval_decks.names)
    eval_bank = jax.tree.map(lambda x: np.tile(np.asarray(x), (ev["repeats"],) + (1,) * (x.ndim - 1)), eval_decks.bank)

    def run_eval(params, key):
        o = evaluate(params, eval_bank, key)
        won = np.asarray(o.won).reshape(ev["repeats"], n_eval)
        res = {"eval_win": float(won.mean()), "eval_truncated": float(np.mean(o.truncated)),
               "eval_errors": int(np.sum(np.asarray(o.error) != 0))}
        res.update({f"eval_win/{name}": float(won[:, i].mean()) for i, name in enumerate(eval_decks.names)})
        return res

    totals = dict(games=0, wins=0, truncated=0, errors=0)
    metrics_file = (out / "metrics.jsonl").open("a")
    last = {}
    for it in range(1, tr["iterations"] + 1):
        t0 = time.perf_counter()
        stats = []
        for _ in range(tr["chunks_per_iteration"]):
            key, k = jax.random.split(key)
            carry, buf, st = play(params, carry, buf, k)
            stats.append(st)
        st = jax.tree.map(lambda *x: int(sum(x)), *jax.device_get(stats))
        t_play = time.perf_counter() - t0
        losses = {}
        if int(buf.size) >= tr["min_replay"]:
            key, k = jax.random.split(key)
            params, opt_state, losses = train(params, opt_state, buf, k)
            losses = {k: float(v) for k, v in jax.device_get(losses).items()}
        for k in totals:
            totals[k] += getattr(st, k)
        last = {
            "iteration": it, "selfplay_games": st.games,
            "selfplay_win_rate": st.wins / st.games if st.games else None,
            "selfplay_win_hp": st.win_hp / st.wins if st.wins else None,
            "decisions_per_s": st.decisions / t_play, "replay_size": int(buf.size),
            **{f"{k}_total": v for k, v in totals.items()}, **losses,
            "seconds": time.perf_counter() - t0,
        }
        if it % ev["every"] == 0 or it == tr["iterations"]:
            key, k = jax.random.split(key)
            last.update(run_eval(params, k))
            state = {"params": params, "opt_state": opt_state}
            (out / "checkpoints" / f"state-{it:05d}.msgpack").write_bytes(serialization.to_bytes(state))
        metrics_file.write(json.dumps(last) + "\n")
        metrics_file.flush()
        log.info("%s", " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in last.items()))

    (out / "params.msgpack").write_bytes(serialization.to_bytes(params))
    summary = {"parameters": int(n_params), "jaxsim_contract": jaxsim.CONTRACT, "devices": [str(d) for d in jax.devices()],
               "train_decks": len(train_decks.names), "train_rejected": len(train_decks.rejected),
               **{k: v for k, v in last.items() if k.startswith("eval_") or k.endswith("_total")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 1 if totals["errors"] else 0


if __name__ == "__main__":
    main(run)
