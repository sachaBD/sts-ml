#!/usr/bin/env python3
"""Gauntlet card-reward evaluation (experiments/act-1-card-selection/approaches/gauntlet).

Reward states: the card reward before each sampled fight of [run] query (fight_index >= 1; bootstrap runs, whose
stored fights replay the run up to it). Per reward state, every option (each offered card, skip) runs the same
gauntlet (apps/gauntlet/worker.cpp):
  elites: for each of the nearest `elite_distances` distinct reachable-elite distances d (combats before the elite),
          d SimpleAgent picks from fresh card rewards, then Gremlin Nob, Lagavulin and Three Sentries;
  boss:   boss_picks picks (fewest combats before the boss), then the act 1 boss;
each repeated for `samples` seeds. Every option shares each seed (same future offers and fight RNG streams).
All fights start at the reward state's HP, relics and potions.

Score per fight (lower is better): HP lost (a death loses all starting HP) + death_penalty if died.
Option score: (1 - boss_weight) * mean elite score + boss_weight * mean boss score (boss only without elites).

out/part-<episode_id>.parquet: one row per fight (episode_id = the fight after the reward).
summary.json: per reward state, each option's scores, the gauntlet's and SimpleAgent's (baseline) choice, and
the paired difference to the baseline option (± 1 standard error).
"""
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from apps.common.app import (TEACHER_KEYS, check_keys, exactly_when, main, required, run_json, snapshot,
                             teacher_settings, value_run, write_json)
from apps.common.replay import decision_rows, diverged, replay_requests
from apps.common.worker import run_parallel, run_worker
from runs import query

log = logging.getLogger(__name__)
SCHEMA = "gauntlet_v1"
BUILT = Path("build/main/gauntlet_worker")  # built by apps/common/job.sh; each run plays with its own copy in out/
RUN_KEYS = {"id", "query", "rewards", "sample_seed", "samples", "elite_distances", "death_penalty", "boss_weight",
            "workers", "value_run", "transition_run", "elite_simulations", "boss_simulations", *TEACHER_KEYS}
NET_LEAVES = ("value_net", "hybrid")  # need [run] value_run
ELITES = ("gremlin_nob", "lagavulin", "three_sentries")
COLUMNS = ["run_seed", "fight_index", "episode_id", "decision_index", "chosen_action", "ascension"]


def inputs(config):
    """The source runs of the queried fights (bootstrap runs), then the value run if any."""
    run = config["run"]
    sources = query.run_ids(run["query"])
    if not sources:
        sys.exit(f"no fights: {run['query']}")
    for source in sources:
        if run_json(source)["inputs"]:
            sys.exit(f"{source}: not a bootstrap run (it has inputs); only bootstrap fights replay")
    return sources + ([run["value_run"]] if "value_run" in run else []) + \
        ([run["transition_run"]] if "transition_run" in run else [])


def seed_of(*parts):
    """A stable 56-bit seed from the parts."""
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:7], "little")


def tasks_for(episode, request, info, distances, samples, simulations_by_kind=None):
    """(episode, worker fight request) per gauntlet fight of one reward state. simulations_by_kind: optional
    {"elite": n, "boss": n} overriding the teacher's simulations for that kind of fight."""
    targets = [(elite, d, "elite") for d in info["elite_distances"][:distances] for elite in ELITES]
    targets.append((info["boss"], info["boss_picks"], "boss"))
    by_kind = simulations_by_kind or {}

    def teacher(kind):
        return {**request["teacher"], "simulations": by_kind[kind]} if kind in by_kind else request["teacher"]
    return [(episode, {**request, "teacher": teacher(kind), "mode": "fight", "encounter": encounter, "picks": picks,
                       "seed": seed_of(request["run_seed"], request["fight_index"], encounter, picks, s)}, kind, s)
            for encounter, picks, kind in targets for s in range(samples)]


def fight_rows(episode, info, task, result, death_penalty):
    """One row per option of a finished gauntlet fight."""
    _, request, kind, sample = task
    rows = []
    for r in result["results"]:
        option = r["option"]
        rows.append({"episode_id": episode, "run_seed": request["run_seed"], "fight_index": request["fight_index"],
                     "floor": info["floor"], "option": option, "card": info["offer"][option] if option >= 0 else "skip",
                     "baseline": option == info["baseline"], "encounter": request["encounter"], "kind": kind,
                     "picks": request["picks"], "sample": sample, "seed": request["seed"], "won": r["won"],
                     "start_hp": r["start_hp"], "final_hp": r["final_hp"],
                     "score": r["start_hp"] - r["final_hp"] + (0 if r["won"] else death_penalty),
                     "picked": r["picked"], "deck_size": r["deck_size"], "seconds": r["seconds"],
                     "simulations": request["teacher"]["simulations"],
                     "pre": json.dumps(r["pre"]) if "pre" in r else None})
    return rows


def mean_se(values):
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    return mean, (math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1) / n) if n > 1 else None)


def evaluate(info, rows, boss_weight):
    """Per option: elite / boss / combined score, deaths, and the paired difference to the baseline option."""
    by = defaultdict(dict)  # option -> (encounter, picks, sample) -> row
    for row in rows:
        by[row["option"]][row["encounter"], row["picks"], row["sample"]] = row
    options = sorted(by, key=lambda o: (o < 0, o))
    has_elites = any(row["kind"] == "elite" for row in rows)
    weight = boss_weight if has_elites else 1.0

    def combine(elite, boss):
        return (1 - weight) * (elite or 0.0) + weight * boss

    result = {}
    for option in options:
        fights = by[option]
        elite = mean_se([r["score"] for r in fights.values() if r["kind"] == "elite"])[0]
        boss = mean_se([r["score"] for r in fights.values() if r["kind"] == "boss"])[0]
        entry = {"card": info["offer"][option] if option >= 0 else "skip", "score": combine(elite, boss),
                 "elite_score": elite, "boss_score": boss,
                 "elite_deaths": sum(not r["won"] for r in fights.values() if r["kind"] == "elite"),
                 "boss_deaths": sum(not r["won"] for r in fights.values() if r["kind"] == "boss")}
        base = by[info["baseline"]]
        diffs = {kind: [fights[k]["score"] - base[k]["score"] for k in fights if fights[k]["kind"] == kind]
                 for kind in ("elite", "boss")}
        (de, se_e), (db, se_b) = mean_se(diffs["elite"]), mean_se(diffs["boss"])
        entry["vs_baseline"] = combine(de, db)
        if se_b is not None and (se_e is not None or not has_elites):
            entry["vs_baseline_se"] = math.sqrt(((1 - weight) * (se_e or 0.0)) ** 2 + (weight * se_b) ** 2)
        result[option] = entry
    choice = min(result, key=lambda o: result[o]["score"])
    return {"choice": choice, "choice_card": result[choice]["card"], "agrees": choice == info["baseline"],
            "options": {str(o): v for o, v in result.items()}}


def write_rows(out, episode, rows):
    tmp = out / f".part-{episode}.tmp"
    pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd")
    os.replace(tmp, out / f"part-{episode}.parquet")


def gauntlet(config, config_path, out):
    run = config["run"]
    check_keys(run, RUN_KEYS, "run")
    teacher = teacher_settings(run)
    if teacher["oracle"] or teacher["random_move"]:
        sys.exit("the gauntlet needs oracle = false and random_move = false")
    exactly_when(run, "value_run", teacher["leaf"] in NET_LEAVES, f"with leaf {' / '.join(NET_LEAVES)}", "run")
    count = required(run, "rewards", "run", int)
    samples = required(run, "samples", "run", int)
    distances = required(run, "elite_distances", "run", int)
    death_penalty = required(run, "death_penalty", "run", float)
    boss_weight = required(run, "boss_weight", "run", float)
    workers = required(run, "workers", "run", int)
    if count < 1 or samples < 1 or distances < 0 or not 0 <= boss_weight <= 1:
        sys.exit("need rewards >= 1, samples >= 1, elite_distances >= 0, 0 <= boss_weight <= 1")
    sources = [r for r in inputs(config) if r not in (run.get("value_run"), run.get("transition_run"))]
    weights = snapshot(value_run(run["value_run"]).weights, out, "value_weights.bin")[0] if "value_run" in run else None
    binary, worker_sha256 = snapshot(BUILT, out)

    fights = query.rows(run["query"], ["episode_id", "fight_index"], "fight_index >= 1", oracle=False)
    candidates = sorted({r["episode_id"] for r in fights})
    if "transition_run" in run:  # only reward states whose earlier fights all replayed exactly (combat_transition_v1)
        schema, date, rid = run["transition_run"].split("/")
        ok = defaultdict(dict)
        parts = sorted((query.RUNS / f"schema={schema}" / f"date={date}" / f"id={rid}" / "out").glob("*.parquet"))
        for t in (x for p in parts for x in pq.read_table(p, columns=["run_seed", "fight_index", "replay"]).to_pylist()):
            ok[t["run_seed"]][t["fight_index"]] = t["replay"] == "ok"
        before = len(candidates)
        candidates = [e for e in candidates if all(ok[e // 100].get(i, False) for i in range(e % 100))]
        log.info("transition_run %s: %d of %d reward states have every earlier fight replayed exactly",
                 run["transition_run"], len(candidates), before)
    episodes = sorted(random.Random(required(run, "sample_seed", "run", int)).sample(candidates, min(count, len(candidates))))
    requests = {e: request for e, (request, _) in
                replay_requests(decision_rows(sources, COLUMNS, {e // 100 for e in episodes}), episodes).items()}
    log.info("%d reward states (of %d candidates), %d samples, %d elite distances, teacher %s, %d workers -> %s",
             len(episodes), len(candidates), samples, distances, teacher, workers, out)

    infos, skipped = {}, []

    def describe(e):  # a source run that no longer replays to the reward state (simulator drift) is skipped
        try:
            return run_worker(binary, {**requests[e], "mode": "describe"})
        except RuntimeError as error:
            if not diverged(error):
                raise
            log.warning("episode %d skipped: replay diverged: %s", e, str(error)[-200:])
            skipped.append(e)
            return {"card_reward": False}
    run_parallel(describe, episodes, workers, lambda e, info: infos.__setitem__(e, info))
    states = [e for e in episodes if infos[e]["card_reward"]]
    for e in states:
        requests[e]["teacher"] = teacher
    by_kind = {k: required(run, f"{k}_simulations", "run", int) for k in ("elite", "boss") if f"{k}_simulations" in run}
    tasks = [t for e in states for t in tasks_for(e, requests[e], infos[e], distances, samples, by_kind)]
    log.info("%d reward states with a card choice; %d gauntlet fights x options", len(states), len(tasks))

    pending = defaultdict(int)
    for task in tasks:
        pending[task[0]] += 1
    rows, summary_states, settings = defaultdict(list), [], None
    started = time.monotonic()

    def on_result(task, result):
        nonlocal settings
        episode = task[0]
        settings = result["teacher"]
        rows[episode].extend(fight_rows(episode, infos[episode], task, result, death_penalty))
        pending[episode] -= 1
        if pending[episode]:
            return
        write_rows(out, episode, rows[episode])
        info = infos[episode]
        state = {"episode_id": episode, **{k: info[k] for k in ("floor", "hp", "max_hp", "boss", "offer", "baseline",
                                                                "elite_distances", "boss_picks", "deck")},
                 **evaluate(info, rows.pop(episode), boss_weight)}
        summary_states.append(state)
        scores = " ".join(f"{v['card']} {v['score']:.1f}" for v in state["options"].values())
        log.info("[%d/%d] floor %d hp %d: %s | gauntlet %s, baseline %s", len(summary_states), len(states),
                 info["floor"], info["hp"], scores, state["choice_card"],
                 info["offer"][info["baseline"]] if info["baseline"] >= 0 else "skip")

    run_parallel(lambda task: run_worker(binary, task[1], weights), tasks, workers, on_result)
    agree = sum(s["agrees"] for s in summary_states)
    write_json(out / "summary.json", {
        "schema": SCHEMA, "query": run["query"], "sources": sources, "value_run": run.get("value_run"),
        "teacher": settings, "worker_sha256": worker_sha256, "samples": samples, "elite_distances": distances,
        "death_penalty": death_penalty, "boss_weight": boss_weight, "reward_states": len(summary_states),
        "simulations_by_kind": by_kind, "skipped_diverged": sorted(skipped),
        "agrees_with_baseline": agree, "minutes": (time.monotonic() - started) / 60,
        "states": sorted(summary_states, key=lambda s: s["episode_id"])})
    log.info("done in %.1f min: gauntlet agrees with SimpleAgent on %d/%d reward states",
             (time.monotonic() - started) / 60, agree, len(summary_states))


if __name__ == "__main__":
    main(gauntlet, inputs)
