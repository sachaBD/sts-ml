#!/usr/bin/env python3
"""Head-to-head test of agents on fresh, never-seen starts of fixed combat [[deck]]s.

Each [[deck]] gets `fights` new 64-bit fight seeds, hashed from this run's id (a collision with any earlier
seed is ~1e-11 likely); `seed_namespace` (default: the run id) regenerates an earlier test's seeds. Each [[agent]]
plays the same starts of its decks: kind "mcts" (pv_worker teacher), "network" (model + PUCT search, greedy,
optional rollout_mix) or "results" (reuse an earlier test's play directory for these exact starts; no play).
Reports win rates per deck and every pairwise paired gap. Errors/timeouts are never losses.
"""
import json
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq

from apps.combat_expert_iteration.expert_iteration import check_decks, load_base, make_starts, play, score
from apps.common.app import check_keys, main, required, sha256, snapshot, write_json
from runs.run import run_dir

log = logging.getLogger(__name__)
RUN_KEYS = {"id", "worker_run", "workers", "fights", "seed_namespace"}
AGENT_KEYS = {"name", "kind", "sims", "model", "rollout_mix", "decks", "results"}


def settings(config):
    run = config["run"]
    check_keys(run, RUN_KEYS, "run")
    for key in ("id", "worker_run"):
        required(run, key, "run", str)
    for key in ("workers", "fights"):
        required(run, key, "run", int)
    decks = config.get("deck") or sys.exit("at least one [[deck]] required")
    check_decks(decks)
    agents = config.get("agent") or sys.exit("at least one [[agent]] required")
    names = {d["name"] for d in decks}
    for a in agents:
        check_keys(a, AGENT_KEYS, "agent")
        required(a, "name", "agent", str)
        kind = required(a, "kind", "agent", str)
        allowed = {"network": {"sims", "model", "rollout_mix"}, "mcts": {"sims"}, "results": {"results"}}
        if kind not in allowed:
            sys.exit(f"agent {a['name']}: kind must be mcts, network or results")
        for key in AGENT_KEYS - {"name", "kind", "decks"}:
            if (key in a) != (key in allowed[kind]):
                sys.exit(f"agent {a['name']}: kind {kind} takes exactly {sorted(allowed[kind])} besides name/kind/decks")
        for key in allowed[kind]:
            required(a, key, "agent", {"sims": int, "rollout_mix": float}.get(key, str))
        if not set(required(a, "decks", "agent", list)) <= names:
            sys.exit(f"agent {a['name']}: unknown decks {set(a['decks']) - names}")
    if len({a["name"] for a in agents}) != len(agents):
        sys.exit("[[agent]] names must be unique")
    return run, decks, agents


def inputs(config):
    return sorted({d["starts"] for d in config["deck"]}) + [config["run"]["worker_run"]]


def reuse(directory, rows):
    """Earlier results for exactly these starts: same fight ids, same seeds, every one present."""
    results = {}
    for line in (Path(directory) / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        results[r["fight_id"]] = r
    for row in rows:
        r = results.get(row["fight_id"]) or sys.exit(f"{directory}: no result for {row['fight_id']}")
        if r["status"] == "completed" and r["fight"]["start"]["seed"] != row["start"]["seed"]:
            sys.exit(f"{directory}: {row['fight_id']} was played from a different seed")
    return {row["fight_id"]: results[row["fight_id"]] for row in rows}


def report(out, run, decks, agents, results):
    lines = [f"# Combat agent test: {run['id']}", "",
             f"{run['fights']} fresh fight seeds per deck, the same starts for every agent. "
             "Errors/timeouts are excluded from pairs, never losses.", ""]
    for d in decks:
        lines += [f"## {d['name']}: deck `{d['deck_id']}`, {d['hp']}/{d['max_hp']} HP", "",
                  "| agent | wins/n | 95% Wilson |", "|---|---:|---:|"]
        players = [a for a in agents if d["name"] in a["decks"]]
        for a in players:
            r = score(results[a["name"]], results[a["name"]], d["name"])  # self-pair: plain win count
            lines.append(f"| {a['name']} | {r['wins']}/{r['n']} | {100 * r['wilson_95'][0]:.1f}-{100 * r['wilson_95'][1]:.1f}% |")
        lines += ["", "Paired gap = row agent minus column agent, percentage points, approx. 95% interval.", "",
                  "| row - column | " + " | ".join(a["name"] for a in players) + " |", "|---|" + "---:|" * len(players)]
        for a in players:
            cells = []
            for b in players:
                if a is b:
                    cells.append("")
                    continue
                r = score(results[b["name"]], results[a["name"]], d["name"])
                half = 1.96 * (r["gap_se"] or 0)
                cells.append(f"{100 * r['gap']:+.1f} [{100 * (r['gap'] - half):+.1f}, {100 * (r['gap'] + half):+.1f}]")
            lines.append(f"| {a['name']} | " + " | ".join(cells) + " |")
        lines.append("")
    (out / "REPORT.md").write_text("\n".join(lines))


def evaluate(config, config_path, out):
    run, decks, agents = settings(config)
    snapshot(config_path, out, "config.toml")
    worker, worker_sha = snapshot(run_dir(run["worker_run"]) / "out/frozen/pv_worker", out)
    used, starts = set(), {}
    for d in decks:
        base, source = load_base(d)
        used |= source
        namespace = run.get("seed_namespace", run["id"])
        starts[d["name"]] = make_starts(base, f"{namespace}:{d['name']}", "test", run["fights"], used)
    write_json(out / "starts.json", starts)
    results, summary = {}, dict(worker_sha256=worker_sha, agents={})
    for a in agents:
        rows = [r for name in a["decks"] for r in starts[name]]
        if a["kind"] == "results":
            results[a["name"]] = reuse(a["results"], rows)
            summary["agents"][a["name"]] = dict(reused=a["results"])
            log.info("%s: reused %d results from %s", a["name"], len(rows), a["results"])
            report(out, run, decks, agents[:agents.index(a) + 1], results)
            continue
        if a["kind"] == "mcts":
            command = [str(worker), "teacher", str(a["sims"])]
        else:
            model = Path(a["model"]).resolve()
            command = [str(worker), "play", str(model / "model.onnx"), str(a["sims"])]
            command += ["--rollout-mix", str(a["rollout_mix"])] if a["rollout_mix"] else []
            summary["agents"][a["name"]] = dict(model_sha256={p.name: sha256(p) for p in sorted(model.glob("model.*"))})
        log.info("%s plays %d fights", a["name"], len(rows))
        results[a["name"]] = play(out / a["name"], rows, command, run["workers"])
        report(out, run, decks, agents[:agents.index(a) + 1], results)
    write_json(out / "summary.json", summary)


if __name__ == "__main__":
    main(evaluate, inputs)
