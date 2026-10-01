#!/usr/bin/env python3
"""card_marginals: paired card-addition fights on synthetic pre-combat states (schema card_marginals_v1).

Supervision for the pre-combat outcome model (docs/research/card-selection/): the marginal effect of adding one card to a
fixed deck. Per deck group (apps/card_marginals/worker.cpp builds it with the simulator from its group_seed): the base
deck ("skip") and the base deck + each of n_candidates candidate cards all play the same fights: every encounter of
the group x `seeds` fight seeds (the same fight RNG streams for every variant).
  stage easy: act 1 easy-pool fights at the run's first 0-2 prior fights (Neow by SimpleAgent, natural card / potion
    rewards); HP drawn per group from natural pre-fight HP of easy fights with fight_index = prior_fights.
  stage hard_elite / boss: counts (prior fights, elites before, upgrades, removed Strikes / Defends, extra relics,
    potions, HP fraction, floor) copied from a donor: a natural pre-fight state of the stage (hard / elite fights, or
    boss fights), uniform over them; the contents come from the simulator (worker.cpp header). Persistent relic
    counters (RELIC_COUNTERS) get a value drawn from the same relic's natural states of the stage.
Natural states: [run] natural_run (combat_transition_v1, replay ok, no random move, buckets natural_buckets).
Encounters per group: [run] encounters (all) + random_per_group drawn from random_encounters (per group).
Budgets: [run] simulations (easy / hard fights), elite_simulations (elite fights), boss_simulations (boss fights).
Generator: [generator].
Groups: group_seed = hash(generator_seed, stage, group); bucket = group_seed % 10 (a later split key; no training here).

  ./apps/card_marginals/run.sh CONFIG.toml [--scratch]      the fights -> out/part-<group>.parquet, summary.json
                                                            ([run] groups = 0: until Ctrl+C, finished groups kept)
  PYTHONPATH=. .venv/bin/python apps/card_marginals/card_marginals.py CONFIG.toml --decks N
                                                            print N groups' decks (no fights; build/main worker)

Row per fight: group, group_seed, bucket, stage, prior_fights, variant (0 = base), card, source (base / reward /
uniform), encounter, kind (easy / hard / elite / boss), seed_index, seed, won, start_hp, battle_final_hp (in-battle
HP before exitBattle; 0 if lost), pre / post (persistent state before BattleContext::init / after exitBattle, as combat_transition_v1), simulations,
seconds, donor_run_seed / donor_fight_index (hard_elite / boss), group_json (the group's description).
"""
import hashlib
import itertools
import json
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from apps.common.app import (TEACHER_KEYS, check_keys, exactly_when, main, required, snapshot, teacher_settings,
                             write_json)
from apps.common.worker import run_parallel, run_worker
from runs.run import RUNS

log = logging.getLogger(__name__)
SCHEMA = "card_marginals_v1"
BUILT = Path("build/main/card_marginals_worker")
RUN_KEYS = {"id", "stage", "groups", "generator_seed", "workers", "encounters", "random_encounters", "random_per_group",
            "seeds", "natural_run", "natural_buckets", "elite_simulations", "boss_simulations", *TEACHER_KEYS}
GENERATOR_KEYS = {"prior_fights_weights", "simple_pick_prob", "skip_prob", "uniform_frac", "n_candidates"}
KINDS = {"easy": ["cultist", "jaw_worm", "two_louse", "small_slimes"],
         "hard": ["gremlin_gang", "lots_of_slimes", "red_slaver", "exordium_thugs", "exordium_wildlife", "blue_slaver",
                  "looter", "large_slime", "three_louse", "two_fungi_beasts"],
         "elite": ["gremlin_nob", "lagavulin", "three_sentries"],
         "boss": ["slime_boss", "the_guardian", "hexaghost"]}
KIND = {e: k for k, es in KINDS.items() for e in es}
STAGE_KINDS = {"easy": {"easy"}, "hard_elite": {"hard", "elite"}, "boss": {"boss"}}
DONOR_CATEGORIES = {"hard_elite": ("hard", "elite"), "boss": ("boss",)}  # natural states a stage's donors come from
# relics whose counter persists between fights and matters in combat (BattleContext reads it at init, writes it at exit)
RELIC_COUNTERS = {"pen_nib", "nunchaku", "happy_flower", "ink_bottle", "sundial", "incense_burner"}

_CARD = pa.struct([("card_id", pa.int64()), ("misc", pa.int64()), ("name", pa.string()), ("upgraded", pa.int64())])
STATE = pa.struct([("deck", pa.list_(_CARD)), ("floor", pa.int64()), ("gold", pa.int64()), ("hp", pa.int64()),
                   ("max_hp", pa.int64()), ("potion_capacity", pa.int64()),
                   ("potions", pa.list_(pa.struct([("name", pa.string()), ("potion_id", pa.int64())]))),
                   ("relics", pa.list_(pa.struct([("data", pa.int64()), ("name", pa.string()), ("relic_id", pa.int64())])))])
ROW = pa.schema([("group", pa.int64()), ("group_seed", pa.int64()), ("bucket", pa.int64()), ("stage", pa.string()),
                 ("prior_fights", pa.int64()), ("variant", pa.int64()), ("card", pa.string()), ("source", pa.string()),
                 ("encounter", pa.string()), ("kind", pa.string()), ("seed_index", pa.int64()), ("seed", pa.int64()), ("won", pa.bool_()),
                 ("start_hp", pa.int64()), ("battle_final_hp", pa.int64()), ("pre", STATE), ("post", STATE),
                 ("simulations", pa.int64()), ("seconds", pa.float64()), ("donor_run_seed", pa.int64()),
                 ("donor_fight_index", pa.int64()), ("group_json", pa.string())])


def inputs(config):
    return [config["run"]["natural_run"]]


def seed_of(*parts):
    """A stable 56-bit seed from the parts."""
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:7], "little")


def natural_states(run_id, buckets):
    """Natural pre-fight states (replay ok, no random move) of a combat_transition_v1 run in `buckets`, with
    elites_before (elite fights earlier in the same run)."""
    schema, date, rid = run_id.split("/")
    cols = ["replay", "any_random", "run_seed", "fight_index", "bucket", "category", "pre"]
    rows = [r for p in sorted((RUNS / f"schema={schema}" / f"date={date}" / f"id={rid}" / "out").glob("*.parquet"))
            for r in pq.read_table(p, columns=cols).to_pylist()]
    elites = {(r["run_seed"], r["fight_index"]) for r in rows if r["category"] == "elite"}
    out = [r for r in rows if r["replay"] == "ok" and not r["any_random"] and r["bucket"] in buckets]
    for r in out:
        r["elites_before"] = sum((r["run_seed"], i) in elites for i in range(r["fight_index"]))
    return out


def donor_of(state):
    """The counts a hard_elite group copies from a natural state (worker.cpp)."""
    pre = state["pre"]
    names = [c["name"] for c in pre["deck"]]
    return {"run_seed": state["run_seed"], "fight_index": state["fight_index"], "category": state["category"],
            "elites_before": state["elites_before"], "floor": pre["floor"], "hp": pre["hp"], "max_hp": pre["max_hp"],
            "upgrades": sum(c["upgraded"] > 0 for c in pre["deck"]),
            "removed": max(0, 9 - names.count("strike_red") - names.count("defend_red")),
            "extra_relics": max(0, len(pre["relics"]) - 1 - state["elites_before"]), "potions": len(pre["potions"])}


def settings(config):
    run, gen = config["run"], config.get("generator", {})
    check_keys(run, RUN_KEYS, "run")
    stage = required(run, "stage", "run", str)
    if stage not in STAGE_KINDS:
        sys.exit(f"[run].stage: one of {sorted(STAGE_KINDS)}")
    check_keys(gen, GENERATOR_KEYS if stage == "easy" else GENERATOR_KEYS - {"prior_fights_weights"}, "generator")
    kinds = STAGE_KINDS[stage]
    encounters = required(run, "encounters", "run", list)
    pool = run.get("random_encounters", [])
    if not encounters and not run.get("random_per_group"):
        sys.exit("[run]: no encounters (encounters empty and no random_per_group)")
    if any(KIND.get(e) not in kinds for e in encounters + pool):
        sys.exit(f"[run].encounters / random_encounters: {stage} encounters from {[e for k in kinds for e in KINDS[k]]}")
    if ("random_encounters" in run) != ("random_per_group" in run) or run.get("random_per_group", 0) > len(pool):
        sys.exit("[run].random_encounters and random_per_group go together; random_per_group <= len(random_encounters)")
    states = natural_states(required(run, "natural_run", "run", str), set(required(run, "natural_buckets", "run", list)))
    generator = {**{k: required(gen, k, "generator", float) for k in ("simple_pick_prob", "skip_prob", "uniform_frac")},
                 "n_candidates": required(gen, "n_candidates", "generator", int)}
    if stage == "easy":
        weights = required(gen, "prior_fights_weights", "generator", list)
        hp = [[s["pre"]["hp"] for s in states if s["category"] == "easy" and s["fight_index"] == i] for i in range(len(weights))]
        if not all(hp):
            sys.exit("no natural easy HP for some prior fight count")
        generator |= {"prior_fights_weights": weights, "hp_by_prior": hp}
        donors = None
    else:
        natural = [s for s in states if s["category"] in DONOR_CATEGORIES[stage]]
        donors = [donor_of(s) for s in natural]
        counters = defaultdict(list)  # relic id -> its natural counter values in the donor states
        for s in natural:
            for r in s["pre"]["relics"]:
                if r["name"] in RELIC_COUNTERS:
                    counters[str(r["relic_id"])].append(r["data"])
        generator["relic_counters"] = dict(counters)
    return run, generator, donors


def base_request(run, generator, donors, group):
    group_seed = seed_of(required(run, "generator_seed", "run", int), run["stage"], group)
    request = {"stage": run["stage"], "ascension": 20, "group_seed": group_seed, "generator": generator}
    rng = random.Random(group_seed)
    if donors:
        request["donor"] = rng.choice(donors)
    encounters = list(run["encounters"])
    if "random_encounters" in run:
        encounters += rng.sample(run["random_encounters"], run["random_per_group"])
    return request, encounters


def fight_rows(group, request, result):
    g = result["group"]
    group_json = json.dumps(g, sort_keys=True)
    donor = request.get("donor") or {}
    return [{"group": group, "group_seed": request["group_seed"], "bucket": request["group_seed"] % 10,
             "stage": request["stage"], "prior_fights": g["prior_fights"], **{k: r[k] for k in (
                 "variant", "card", "source", "encounter", "kind", "seed_index", "seed", "won", "start_hp",
                 "battle_final_hp", "pre", "post", "simulations", "seconds")},
             "donor_run_seed": donor.get("run_seed"), "donor_fight_index": donor.get("fight_index"), "group_json": group_json}
            for r in result["results"]]


def generate(config, config_path, out):
    run, generator, donors = settings(config)
    groups = required(run, "groups", "run", int)  # 0 = run until Ctrl+C
    if groups < 0:
        sys.exit("[run] groups: need >= 0 (0 = until interrupted)")
    workers = required(run, "workers", "run", int)
    seeds = required(run, "seeds", "run", int)
    teacher = teacher_settings(run)
    if teacher["oracle"] or teacher["random_move"]:
        sys.exit("card_marginals needs oracle = false and random_move = false")
    if teacher["leaf"] != "guided_rollout":
        sys.exit("card_marginals: leaf guided_rollout only (no value run input)")
    kinds = {KIND[e] for e in run["encounters"] + run.get("random_encounters", [])}
    for kind in ("elite", "boss"):  # [run] simulations: easy / hard fights; these override it for their kind
        exactly_when(run, f"{kind}_simulations", kind in kinds, f"with {kind} encounters", "run")
    teachers = {k: {**teacher, "simulations": run[f"{k}_simulations"]} if k in ("elite", "boss") else teacher
                for k in kinds}
    binary, worker_sha256 = snapshot(BUILT, out)
    per_group = len(run["encounters"]) + run.get("random_per_group", 0)
    log.info("%d groups x (1 + %d variants) x %d encounters x %d seeds, teachers %s, %d workers -> %s", groups,
             generator["n_candidates"], per_group, seeds, teachers, workers, out)
    started, counts, done, settings_seen = time.monotonic(), Counter(), [0], None

    def request(group):
        base, encounters = base_request(run, generator, donors, group)
        return {**base, "mode": "fight", "encounters": encounters, "seeds": seeds, "teachers": teachers}

    def on_result(group, result):
        nonlocal settings_seen
        settings_seen = result["teachers"]
        rows = fight_rows(group, request(group), result)
        tmp = out / f".part-{group:06d}.tmp"
        pq.write_table(pa.Table.from_pylist(rows, schema=ROW), tmp, compression="zstd")
        tmp.replace(out / f"part-{group:06d}.parquet")
        counts["fights"] += len(rows)
        counts["losses"] += sum(not r["won"] for r in rows)
        done[0] += 1
        if done[0] % (max(1, groups // 20) if groups else 10) == 0 or done[0] == groups:
            log.info("[%d/%s] groups, %d fights, %d losses, %.1f min", done[0], groups or "forever", counts["fights"],
                     counts["losses"], (time.monotonic() - started) / 60)

    try:  # groups = 0: until Ctrl+C; finished groups are kept (a part is written atomically per group)
        run_parallel(lambda group: run_worker(binary, request(group)), itertools.count() if groups == 0 else range(groups),
                     workers, on_result)
        interrupted = False
    except KeyboardInterrupt:
        if groups:
            raise
        interrupted = True
        log.info("interrupted: keeping the %d finished groups (groups 0..%d may have gaps: in-flight ones are dropped)",
                 done[0], done[0] + workers - 1)
    write_json(out / "summary.json", {
        "schema": SCHEMA, "stage": run["stage"], "groups": groups or None, "groups_done": done[0], "interrupted": interrupted, "generator_seed": run["generator_seed"],
        "generator": {k: v for k, v in generator.items() if k not in ("hp_by_prior", "relic_counters")},
        "relic_counter_samples": {k: len(v) for k, v in generator.get("relic_counters", {}).items()},
        "natural": {"run": run["natural_run"], "buckets": run["natural_buckets"], "donors": len(donors) if donors else None,
                    "hp_count_by_prior": [len(h) for h in generator["hp_by_prior"]] if "hp_by_prior" in generator else None},
        "encounters": run["encounters"], "random_encounters": run.get("random_encounters"),
        "random_per_group": run.get("random_per_group"), "seeds": seeds, "teachers": settings_seen, "worker_sha256": worker_sha256,
        "fights": counts["fights"], "losses": counts["losses"], "minutes": (time.monotonic() - started) / 60})
    log.info("done: %d fights, %d losses, %.1f min", counts["fights"], counts["losses"], (time.monotonic() - started) / 60)


def print_decks(config_path, count):
    import tomllib
    config = tomllib.loads(Path(config_path).read_text())
    run, generator, donors = settings(config)
    results = []
    for g in range(count):
        request, encounters = base_request(run, generator, donors, g)
        results.append({**run_worker(BUILT, {**request, "mode": "deck"})["group"], "encounters": encounters})
    starter = {"strike_red", "defend_red", "bash", "ascenders_bane"}
    for g, d in enumerate(results):
        counts = Counter(d["deck"])
        basics = " ".join(f"{n}x{c.replace('_red', '')}" for c, n in sorted(counts.items()) if c.rstrip("+") in starter)
        extra = ", ".join(f"{n}x {c}" if n > 1 else c for c, n in sorted(counts.items()) if c.rstrip("+") not in starter)
        picks = "; ".join(f"{'ELITE ' if r.get('room') == 'elite' else ''}[{', '.join(r['offer'])}] -> {r['pick']} ({r['by']})"
                          for r in d["rewards"]) or "-"
        print(f"group {g}: prior fights {d['prior_fights']}, floor {d['floor']}, hp {d['hp']}/{d['max_hp']}, "
              f"potions {d['potions'] or '-'}")
        if "donor" in d:
            x = d["donor"]
            print(f"  donor: run {x['run_seed']} fight {x['fight_index']} ({x['category']}): elites before {x['elites_before']}, "
                  f"upgrades {x['upgrades']}, removed {x['removed']}, extra relics {x['extra_relics']}, potions {x['potions']}")
        print(f"  neow: {d['neow']['bonus']} {d['neow']['drawback']}".rstrip())
        print(f"  rewards: {picks}")
        if d.get("relics_added") or d.get("removes") or d.get("upgrades"):
            counters = d.get("relic_counters") or {}
            print(f"  relics: {', '.join(r + (f'({counters[r]})' if r in counters else '') for r in d['relics'] if r != 'burning_blood') or '-'} | "
                  f"removed: {', '.join(d.get('removes', [])) or '-'} | upgraded: {', '.join(d.get('upgrades', [])) or '-'}")
        print(f"  deck ({len(d['deck'])}): {basics} | {extra or '-'}")
        print(f"  candidates: {', '.join(c['card'] + (' (uniform)' if c['source'] == 'uniform' else '') for c in d['candidates'])}")
        print(f"  encounters: {', '.join(d['encounters'])}")
    summary = {"prior_fights": Counter(d["prior_fights"] for d in results),
               "potions": Counter(len(d["potions"]) for d in results),
               "deck_size": Counter(len(d["deck"]) for d in results),
               "upgraded": Counter(sum(c.endswith("+") for c in d["deck"]) for d in results),
               "removed": Counter(max(0, 9 - sum(c.rstrip("+") in ("strike_red", "defend_red") for c in d["deck"])) for d in results),
               "relics_excl_bb": Counter(len(d["relics"]) - 1 for d in results)}
    print({k: dict(sorted(v.items())) for k, v in summary.items()})


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[2] == "--decks":
        print_decks(sys.argv[1], int(sys.argv[3]))
    else:
        main(generate, inputs)
