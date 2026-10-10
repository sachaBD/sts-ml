#!/usr/bin/env python3
"""Expert iteration on fixed combat starts (deck, HP, relics), one or more [[deck]]s trained jointly into one
network; only the fight RNG seed varies. Fight budgets below are per deck.

1. Bootstrap: MCTS teacher (`pv_worker teacher`) plays `bootstrap_fights`; a fresh policy/value
   network is trained on them. The teacher also plays the `monitor_fights` once, as the reference.
2. Each update: the network + PUCT search (`pv_worker play`, optionally with guided-rollout leaf
   values, `rollout_mix`) plays `batch_fights` new fights; the network is retrained on the last
   `replay_batches` batches plus a tapering slice of the teacher fights.
3. Every `eval_every` updates (and at the end) the network + search plays the monitor fights,
   greedily, and is compared fight by fight with the teacher.

Results are reported per deck and pooled. Value target: 100 x actual win. Policy target: normalized root visits. Monitor fights never train.
rollout_mix = 0 is the network-only recipe; 0.5 is the rollout-guided recipe.
"""
import copy
import re
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from apps.common.app import check_keys, main, required, snapshot, write_json
from apps.common.worker import run_parallel
from apps.human_champ.bench import MASK, play_one, rng_state
from apps.megacrit_dump.schema import load
from runs.run import run_dir

log = logging.getLogger(__name__)
PY = sys.executable
KEYS = {"id", "worker_run", "workers", "device",
        "teacher_sims", "bootstrap_fights", "bootstrap_epochs", "bootstrap_lr", "monitor_fights",
        "updates", "batch_fights", "sims", "rollout_mix", "explore", "sample_turns",
        "epochs", "lr", "replay_batches", "teacher_taper", "states_per_fight", "width", "eval_every"}
DECK_KEYS = {"name", "starts", "deck_id", "hp", "max_hp"}
INTS = ("workers", "teacher_sims", "bootstrap_fights", "bootstrap_epochs", "monitor_fights", "updates",
        "batch_fights", "sims", "epochs", "replay_batches", "teacher_taper", "states_per_fight", "width", "eval_every")
FIGHT_TIMEOUT = 900  # seconds; a timed-out fight is an error, never a loss


def settings(config):
    run = config["run"]
    check_keys(run, KEYS, "run")
    for key in INTS:
        required(run, key, "run", int)
    for key in ("bootstrap_lr", "lr", "rollout_mix"):
        required(run, key, "run", float)
    for key in ("explore", "sample_turns"):
        required(run, key, "run", bool)
    for key in ("id", "worker_run", "device"):
        required(run, key, "run", str)
    if not 0 <= run["rollout_mix"] <= 1:
        sys.exit("[run].rollout_mix must be in [0, 1]")
    decks = config.get("deck") or sys.exit("at least one [[deck]] required")
    check_decks(decks)
    return run, decks


def inputs(config):
    return sorted({d["starts"] for d in config["deck"]}) + [config["run"]["worker_run"]]


def deck_of(fight_id):
    return fight_id.split(":")[1]  # fight ids are run_id:deck:split:index


def wilson(wins, n):
    z = 1.959963984540054
    p, den = wins / n, 1 + z * z / n
    center, half = (p + z * z / (2 * n)) / den, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0., center - half), min(1., center + half)]


def make_starts(base, namespace, split, n, used):
    rows = []
    for i in range(n):
        seed = int.from_bytes(hashlib.sha256(f"{namespace}:{split}:{i}".encode()).digest()[:8], "little")
        while seed in used:
            seed = (seed + 1) & MASK
        used.add(seed)
        r = copy.deepcopy(base)
        r.update(fight_id=f"{namespace}:{split}:{i}", seed_kind=split)
        r["start"].update(seed=seed, misc_rng=rng_state(seed), potion_rng=rng_state(seed))
        rows.append(r)
    return rows


def play(directory, rows, command, workers):
    """Play `rows` (starts) with the worker `command`, journaled to directory/results.jsonl; {fight_id: result}.
    Errors and timeouts are recorded, never counted as losses; more than 5% of them stops the run."""
    directory.mkdir(parents=True)
    write_json(directory / "command.json", command)
    results = {}
    with (directory / "results.jsonl").open("w") as journal:
        def done(_, result):
            journal.write(json.dumps(result) + "\n")
            journal.flush()
            results[result["fight_id"]] = result
        run_parallel(play_one, [(command, row, FIGHT_TIMEOUT) for row in rows], workers, done)
    completed = [x for x in results.values() if x["status"] == "completed"]
    v4 = load("combat_v4")
    pq.write_table(pa.Table.from_pylist([x["fight"] for x in completed], v4.FIGHTS), directory / "fights.parquet")
    pq.write_table(pa.Table.from_pylist([s for x in completed for s in x.get("search", [])], v4.SEARCH),
                   directory / "search.parquet")
    wins = sum(x["fight"]["won"] for x in completed)
    log.info("%s: %d/%d completed, %d wins", directory.name, len(completed), len(rows), wins)
    if len(completed) < .95 * len(rows):
        sys.exit(f"{directory}: more than 5% of fights errored or timed out; see results.jsonl")
    return results


def load_base(deck):
    """The stored human start of a [[deck]], checked against its configured HP and no potions; (start row, source seeds)."""
    source = pq.read_table(deck["starts"]).to_pylist()  # repo-relative starts parquet
    base = next(r for r in source if r["deck_id"] == deck["deck_id"] and r["seed_kind"] == "human")
    start = base["start"]
    if (start["hp"], start["max_hp"]) != (deck["hp"], deck["max_hp"]) or start["potions"] != [1, 1]:
        sys.exit(f"deck {deck['name']} start is {start['hp']}/{start['max_hp']} HP, potions {start['potions']}; "
                 f"config expects {deck['hp']}/{deck['max_hp']} and no potions")
    return base, {r["start"]["seed"] for r in source}


def check_decks(decks):
    for deck in decks:
        check_keys(deck, DECK_KEYS, "deck")
        for key in ("name", "starts", "deck_id"):
            required(deck, key, "deck", str)
        for key in ("hp", "max_hp"):
            required(deck, key, "deck", int)
        if not re.fullmatch(r"[a-z0-9_-]+", deck["name"]):
            sys.exit(f"[[deck]].name {deck['name']!r}: use [a-z0-9_-]")
    if len({d["name"] for d in decks}) != len(decks):
        sys.exit("[[deck]] names must be unique")


class Experiment:
    def __init__(self, run, decks, out):
        self.run, self.out, self.bases, self.used = run, out, {}, set()
        for deck in decks:
            self.bases[deck["name"]], seeds = load_base(deck)
            self.used |= seeds
        self.worker, self.worker_sha = snapshot(run_dir(run["worker_run"]) / "out/frozen/pv_worker", out)
        self.split = {}  # fight_id -> train / val, for the trainer
        self.validation_rows = None

    def starts(self, name, n):
        """n new starts per deck."""
        rows = [r for deck, base in self.bases.items()
                for r in make_starts(base, f"{self.run['id']}:{deck}", name, n, self.used)]
        self.split.update({r["fight_id"]: "val" if name == "monitor" else "train" for r in rows})
        write_json(self.out / "split.json", self.split)
        return rows

    def play(self, directory, rows, model=None, collect=False):
        """Teacher if model is None, else network + search; returns {fight_id: result}."""
        r = self.run
        if model is None:
            command = [str(self.worker), "teacher", str(r["teacher_sims"])]
        else:
            command = [str(self.worker), "play", str(model / "model.onnx"), str(r["sims"])]
            if r["rollout_mix"]:
                command += ["--rollout-mix", str(r["rollout_mix"])]
            if collect:
                command += ["--explore"] * r["explore"] + ["--sample-turns"] * r["sample_turns"]
        return play(directory, rows, command, r["workers"])

    def call(self, directory, name, command):
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (directory / f"{name}.log").open("w") as stream:
            subprocess.run([str(c) for c in command], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def encode(self, directory):
        """Replay the played fights into training rows (features, visit targets, outcome)."""
        self.call(directory, "encode", [PY, "-m", "agents.combat.pv.data", "--fights", directory / "fights.parquet",
                                              "--search", directory / "search.parquet", "--encounters", 39,
                                              "--worker", self.worker, "--out", directory / "rows"])
        return directory / "rows/rows.parquet"

    def train(self, directory, shards, fight_ids, init, epochs, lr):
        r = self.run
        write_json(directory / "train-fights.json", fight_ids)
        # Monitor rows give validation losses only; split.json marks them val, so they never train.
        command = [PY, "-m", "agents.combat.pv.train", "--data", *shards, self.validation_rows, "--split-manifest", self.out / "split.json",
                   "--train-fights", directory / "train-fights.json", "--states-per-fight", r["states_per_fight"],
                   "--flat-policy-weighting", "--width", r["width"], "--epochs", epochs, "--lr", lr, "--grad-clip", 1,
                   "--device", r["device"], "--stream", "--out", directory / "model"]
        command += ["--value-activation", "sigmoid"] if init is None else ["--init", init / "model.pt", "--resume-optimizer"]
        self.call(directory, "train", command)
        return directory / "model"


def score(reference, candidate, deck=None):
    """Paired on monitor fights both agents completed (of `deck`, or all); each fight seed is an independent trial."""
    ids = [k for k in reference if reference[k]["status"] == candidate.get(k, {}).get("status") == "completed"
           and deck in (None, deck_of(k))]
    n = len(ids)
    pairs = [(reference[k]["fight"]["won"], candidate[k]["fight"]["won"]) for k in ids]
    d = [int(b) - int(a) for a, b in pairs]
    gap = sum(d) / n
    se = math.sqrt(sum((x - gap) ** 2 for x in d) / (n * (n - 1))) if n > 1 else None
    wins = sum(b for _, b in pairs)
    return dict(n=n, wins=wins, win_rate=wins / n, wilson_95=wilson(wins, n), teacher_wins=sum(a for a, _ in pairs),
                gap=gap, gap_se=se)


def report(out, run, decks, curve):
    lines = [f"# Combat expert iteration: {run['id']}", ""]
    lines += [f"- {d['name']}: deck `{d['deck_id']}`, {d['hp']}/{d['max_hp']} HP, no potions." for d in decks]
    lines += ["", f"Search {run['sims']} sims, rollout_mix {run['rollout_mix']}. Teacher MCTS {run['teacher_sims']} sims. "
              "One network trained on all decks; training fights are per deck.",
              "Monitor fights are reused at every evaluation: descriptive, not a final test.", "",
              "| update | training fights/deck | deck | wins/n | 95% Wilson | teacher wins | paired gap +-1 SE |",
              "|---|---:|---|---:|---:|---:|---:|"]
    for c in curve:
        for name, r in [*c["decks"].items(), *([("all", c["all"])] if len(c["decks"]) > 1 else [])]:
            lo, hi = r["wilson_95"]
            se = "?" if r["gap_se"] is None else f"{100 * r['gap_se']:.1f}"
            lines.append(f"| {c['update']} | {c['training_fights']} | {name} | {r['wins']}/{r['n']} | {100 * lo:.1f}-{100 * hi:.1f}% | "
                         f"{r['teacher_wins']}/{r['n']} | {100 * r['gap']:+.1f} +-{se} pts |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def expert_iteration(config, config_path, out):
    run, decks = settings(config)
    snapshot(config_path, out, "config.toml")
    x = Experiment(run, decks, out)
    bootstrap, monitor = x.starts("bootstrap", run["bootstrap_fights"]), x.starts("monitor", run["monitor_fights"])
    log.info("bootstrap: teacher plays %d training fights and %d monitor fights", len(bootstrap), len(monitor))
    teacher = x.play(out / "bootstrap/play", bootstrap)
    reference = x.play(out / "monitor-teacher", monitor)
    teacher_rows = x.encode(out / "bootstrap/play")
    x.validation_rows = x.encode(out / "monitor-teacher")
    teacher_ids = {d["name"]: [r["fight_id"] for r in bootstrap if deck_of(r["fight_id"]) == d["name"]
                               and teacher[r["fight_id"]]["status"] == "completed"] for d in decks}
    all_teacher = [k for ks in teacher_ids.values() for k in ks]
    model = x.train(out / "bootstrap", [teacher_rows], all_teacher, None, run["bootstrap_epochs"], run["bootstrap_lr"])

    curve, batches = [], []

    def evaluate(update, directory, teacher_used):
        played = x.play(directory / "monitor", monitor, model)
        result = dict(update=update, training_fights=run["bootstrap_fights"] + update * run["batch_fights"],
                      teacher_replay_fights=teacher_used, model=str(model),
                      decks={d["name"]: score(reference, played, d["name"]) for d in decks}, all=score(reference, played))
        curve.append(result)
        write_json(out / "curve.json", curve)
        report(out, run, decks, curve)
        log.info("update %d monitor: %s", update, {k: f"{v['wins']}/{v['n']}" for k, v in result["decks"].items()})

    evaluate(0, out / "bootstrap", len(all_teacher))
    for update in range(1, run["updates"] + 1):
        directory = out / f"update{update:03d}"
        played = x.play(directory / "play", x.starts(f"learner-{update}", run["batch_fights"]), model, collect=True)
        batches.append((x.encode(directory / "play"), [k for k, v in played.items() if v["status"] == "completed"]))
        replay = batches[-run["replay_batches"]:]
        keep = max(0, run["bootstrap_fights"] - run["teacher_taper"] * (update - 1))
        anchors = [k for ks in teacher_ids.values() for k in ks[:keep]]
        shards = ([teacher_rows] if anchors else []) + [rows for rows, _ in replay]
        ids = anchors + [k for _, ks in replay for k in ks]
        model = x.train(directory, shards, ids, model, run["epochs"], run["lr"])
        if update % run["eval_every"] == 0 or update == run["updates"]:
            evaluate(update, directory, len(anchors))
    write_json(out / "summary.json", dict(decks=decks, rollout_mix=run["rollout_mix"], worker_sha256=x.worker_sha,
                                          latest_model=str(model), curve=curve))


if __name__ == "__main__":
    main(expert_iteration, inputs)
