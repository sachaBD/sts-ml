"""Read-only dashboard for a combat_expert_iteration run; safe while the run is writing.

watch -n 5 '.venv/bin/python -m apps.combat_expert_iteration.status --run RUN_DIR'
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import tomllib


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def results(directory):
    """Rows of a play journal; a half-written last line is skipped."""
    rows = []
    try:
        lines = (directory / "results.jsonl").read_text().splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def play_line(name, directory, total, decks):
    rows = results(directory)
    if rows is None:
        return None
    c = Counter(r["status"] for r in rows)
    wins = sum(r["fight"]["won"] for r in rows if r["status"] == "completed")
    secs = [r["seconds"] for r in rows if r["status"] == "completed"]
    pace = f", {sum(secs) / len(secs):.0f}s/fight" if secs else ""
    bad = ", ".join(f"{n} {k}" for k, n in c.items() if k != "completed")
    done = len(rows) == total
    line = (f"  {'done' if done else 'RUN '} {name:<22} {len(rows):>4}/{total:<4} played, {wins:>4} wins"
            f" ({100 * wins / max(1, c['completed']):.0f}%){pace}{', ' + bad if bad else ''}")
    # Per deck (fight ids are run:deck:split:index): wins/played and win rate.
    n, w = Counter(), Counter()
    for r in rows:
        deck = r["fight_id"].split(":")[1]
        n[deck] += 1
        w[deck] += r["status"] == "completed" and r["fight"]["won"]
    if len(decks) > 1:
        line += "\n" + " " * 30 + "  ".join(f"{d} {w[d]}/{n[d]} ({100 * w[d] / n[d]:.0f}%)" for d in decks if n[d])
    return line, done


def render(run):
    run = Path(run)
    out = run / "out"
    meta = read_json(run / "run.json") or {}
    try:
        config = tomllib.loads((out / "config.toml").read_text())
        cfg = config["run"]
        # Runs launched before [[deck]] existed kept a single deck in [run].
        decks = config.get("deck") or [dict(name="deck", deck_id=cfg["deck_id"], hp=cfg["hp"], max_hp=cfg["max_hp"])]
    except FileNotFoundError:
        return f"RUN: {run}\nnot started (no out/config.toml)"
    age = time.time() - max((p.stat().st_mtime for p in out.rglob("*")), default=time.time())
    lines = [f"RUN: {meta.get('run_id', run.name)}   status {meta.get('status', '?')}   last write {age:.0f}s ago",
             "Decks: " + ", ".join(f"{d['name']} ({d['hp']}/{d['max_hp']} HP)" for d in decks),
             f"Search {cfg['sims']} sims, rollout_mix {cfg['rollout_mix']}; teacher {cfg['teacher_sims']} sims; {cfg['workers']} workers",
             f"Plan per deck: {cfg['bootstrap_fights']} teacher fights, {cfg['updates']} x {cfg['batch_fights']} learner fights, "
             f"{cfg['monitor_fights']} monitor fights every {cfg['eval_every']} update(s)", "", "STAGES"]
    k = len(decks)  # play stages cover every deck
    stages = [("teacher bootstrap", out / "bootstrap/play", k * cfg["bootstrap_fights"]),
              ("teacher monitor", out / "monitor-teacher", k * cfg["monitor_fights"]),
              ("bootstrap train", out / "bootstrap/model", None),
              ("monitor update 0", out / "bootstrap/monitor", k * cfg["monitor_fights"])]
    for u in range(1, cfg["updates"] + 1):
        d = out / f"update{u:03d}"
        stages += [(f"collect update {u}", d / "play", k * cfg["batch_fights"]), (f"train update {u}", d / "model", None)]
        if u % cfg["eval_every"] == 0 or u == cfg["updates"]:
            stages.append((f"monitor update {u}", d / "monitor", k * cfg["monitor_fights"]))
    for name, directory, total in stages:
        if total is None:
            if (directory / "model.onnx").exists():
                lines.append(f"  done {name}")
            elif directory.parent.joinpath("train.log").exists():
                log = directory.parent.joinpath("train.log").read_text().splitlines()
                epochs = sum(line.startswith('{"epoch"') for line in log)
                lines.append(f"  RUN  {name:<22} {epochs} epoch(s) logged")
            continue
        line = play_line(name, directory, total, [d["name"] for d in decks])
        if line:
            lines.append(line[0])
        elif directory.parent.exists() and name.startswith("collect"):
            lines.append(f"  RUN  {name:<22} encoding / starting")
    curve = [c if "decks" in c else dict(c, decks={decks[0]["name"]: c}) for c in read_json(out / "curve.json") or []]
    if curve:
        lines += ["", "MONITOR vs teacher (same seeds; reused every evaluation, descriptive only)",
                  "  update  fights/deck  deck         learner          95% Wilson    teacher   gap +-1 SE"]
        for c in curve:
            for name, r in [*c["decks"].items(), *([("all", c["all"])] if len(c["decks"]) > 1 else [])]:
                lo, hi = r["wilson_95"]
                se = "?" if r["gap_se"] is None else f"{100 * r['gap_se']:.1f}"
                lines.append(f"  {c['update']:>6}  {c['training_fights']:>11}  {name:<11} {r['wins']:>3}/{r['n']:<3} {100 * r['win_rate']:5.1f}%"
                             f"   {100 * lo:4.1f}-{100 * hi:5.1f}%   {r['teacher_wins']:>3}/{r['n']:<3} {100 * r['gap']:+5.1f} +-{se} pts")
    if meta.get("status") == "failed":
        lines += ["", f"FAILED: see {run / 'logs/stderr.log'}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="runs/schema=combat_v4/date=.../id=...")
    print(render(p.parse_args().run))


if __name__ == "__main__":
    main()
