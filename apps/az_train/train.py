#!/usr/bin/env python3
"""AlphaZero-style self-play training on the JAX simulator: Ironclad vs The Champ, A20 (agents/combat/az).

Every setting is explicit in the TOML (no defaults):
  [run]       id
  [decks]     training start bank: agents.combat.az.decks.DeckConfig
  [model]     agents.combat.az.model.ModelConfig
  [search]    agents.combat.az.search.SearchConfig for self-play
  [selfplay]  games (parallel), chunk (decisions per jitted call)
  [train]     iterations, chunks_per_iteration, steps_per_iteration, batch_size, lr, weight_decay, value_weight,
              replay_capacity, min_replay, seed
  [eval]      every (iterations), simulations, particles, chunk, decks (a DeckConfig table), hp (list: one
              evaluation per starting hp, every eval deck)
Writes to out/: config.json, metrics.jsonl (one line per iteration), checkpoints/params-<iteration>.msgpack,
params.msgpack (final), summary.json.
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
SELFPLAY = {"games": int, "chunk": int}
TRAIN = {"iterations": int, "chunks_per_iteration": int, "steps_per_iteration": int, "batch_size": int, "lr": float,
         "weight_decay": float, "value_weight": float, "replay_capacity": int, "min_replay": int, "seed": int}
EVAL = {"every": int, "simulations": int, "particles": int, "chunk": int, "decks": dict, "hp": list}


def typed(table: dict, spec: dict, name: str) -> dict:
    check_keys(table, spec, name)
    return {k: required(table, k, name, kind) for k, kind in spec.items()}


def dataclass_from(cls, table: dict, name: str):
    fields = {f.name: f.type for f in dataclasses.fields(cls)}
    kinds = {"int": int, "float": float}
    return cls(**typed(table, {k: kinds[t] if isinstance(t, str) else t for k, t in fields.items()}, name))


def parse(config: dict[str, Any]):
    from agents.combat.az import decks, model, search
    check_keys(config, TABLES, "config")
    for t in TABLES:
        required(config, t, "config", dict)
    ev = typed(config["eval"], EVAL, "eval")
    return dict(
        decks=dataclass_from(decks.DeckConfig, config["decks"], "decks"),
        model=dataclass_from(model.ModelConfig, config["model"], "model"),
        search=dataclass_from(search.SearchConfig, config["search"], "search"),
        selfplay=typed(config["selfplay"], SELFPLAY, "selfplay"),
        train=typed(config["train"], TRAIN, "train"),
        eval={**ev, "decks": dataclass_from(decks.DeckConfig, ev["decks"], "eval.decks")},
    )


def run(config: dict[str, Any], config_path: Path, out: Path) -> int:
    import jax
    import numpy as np
    from flax import serialization

    import jaxsim
    from agents.combat.az import decks, learn, model, replay, search, selfplay

    c = parse(config)
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    log.info("devices %s, jaxsim %s", jax.devices(), jaxsim.CONTRACT)

    spec = jaxsim.DEFAULT_SPEC
    env = jaxsim.make_env(spec)
    net = model.AZNet(spec, c["model"])
    bank = decks.make_bank(c["decks"], spec)
    tr = c["train"]
    key = jax.random.key(tr["seed"])
    key, k_init = jax.random.split(key)
    s0 = env.init(jax.tree.map(lambda x: x[0], bank), k_init)
    params = net.init(k_init, env.observe(s0))
    n_params = sum(x.size for x in jax.tree.leaves(params))
    log.info("parameters %d", n_params)

    tcfg = learn.TrainConfig(tr["batch_size"], tr["lr"], tr["weight_decay"], tr["value_weight"])
    opt = learn.make_optimizer(tcfg)
    opt_state = opt.init(params)
    train_step = learn.make_train_step(net, opt, tcfg)
    policy = search.make_policy(env, net, c["search"])
    sp = c["selfplay"]
    start, play = selfplay.make_selfplay(env, policy, bank, sp["games"], sp["chunk"], explore=True)
    key, k = jax.random.split(key)
    carry = start(k)
    buf = replay.Replay(tr["replay_capacity"], tr["seed"])

    ev = c["eval"]
    eval_policy = search.make_policy(env, net, search.SearchConfig(ev["simulations"], ev["particles"],
                                                                    c["search"].max_considered, 0.0))
    evaluate = selfplay.make_eval(env, eval_policy, ev["chunk"])
    eval_decks = decks.make_bank(ev["decks"], spec)

    def run_eval(params, key):
        res = {}
        for hp in ev["hp"]:
            b = eval_decks._replace(hp=eval_decks.hp * 0 + hp, max_hp=np.maximum(eval_decks.max_hp * 0 + hp, 80))
            o = evaluate(params, b, key)
            res[f"eval_win_hp{hp}"] = float(np.mean(o.won))
            res[f"eval_truncated_hp{hp}"] = float(np.mean(o.truncated))
        return res

    metrics_file = (out / "metrics.jsonl").open("a")
    last = {}
    for it in range(1, tr["iterations"] + 1):
        t0 = time.perf_counter()
        g0, w0 = buf.games, buf.wins
        for _ in range(tr["chunks_per_iteration"]):
            key, k = jax.random.split(key)
            carry, rec = play(params, carry, k)
            buf.add(rec)
        t_play = time.perf_counter() - t0
        losses = []
        if buf.size >= tr["min_replay"]:
            for _ in range(tr["steps_per_iteration"]):
                b = buf.sample(tr["batch_size"])
                params, opt_state, m = train_step(params, opt_state, replay.batch_obs(b), b["mask"], b["policy"], b["value"])
                losses.append(m)
        losses = jax.device_get(losses)
        decisions = tr["chunks_per_iteration"] * sp["chunk"] * sp["games"]
        games = buf.games - g0
        last = {
            "iteration": it, "selfplay_games": games,
            "selfplay_win_rate": (buf.wins - w0) / games if games else None,
            "decisions_per_s": decisions / t_play, "replay_size": buf.size, "games_total": buf.games,
            "truncated_total": buf.truncated, "errors_total": buf.errors,
            **({k: float(np.mean([m[k] for m in losses])) for k in losses[0]} if losses else {}),
            "seconds": time.perf_counter() - t0,
        }
        if it % ev["every"] == 0 or it == tr["iterations"]:
            key, k = jax.random.split(key)
            last.update(run_eval(params, k))
            (out / "checkpoints" / f"params-{it:05d}.msgpack").write_bytes(serialization.to_bytes(params))
        metrics_file.write(json.dumps(last) + "\n")
        metrics_file.flush()
        log.info("%s", " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in last.items()))

    (out / "params.msgpack").write_bytes(serialization.to_bytes(params))
    summary = {"parameters": int(n_params), "jaxsim_contract": jaxsim.CONTRACT, "devices": [str(d) for d in jax.devices()],
               **{k: v for k, v in last.items() if k.startswith("eval_") or k in ("games_total", "errors_total")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 1 if buf.errors else 0


if __name__ == "__main__":
    main(run)
