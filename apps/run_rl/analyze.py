#!/usr/bin/env python3
"""Concise run diagnostics; optional seed-paired comparison (DIR2 minus DIR)."""
import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.overworld.value.core import acts_cleared, read_runs, run_score


def mean_se(values):
    """Sample mean and standard error; SE is unknown for fewer than two samples."""
    if not values:
        return "n/a (n=0)"
    se = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return f"{statistics.mean(values):.3f} ± {se:.3f} (n={len(values)})" if se is not None else f"{values[0]:.3f} ± n/a (n=1)"


def distribution(counts):
    return ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "none"


def fight_act(step):
    state = step.get("state", {})
    return state.get("act", 1 + max(0, state.get("floor", 0) - 1) // 17)


def summary(runs, label):
    deaths, fights, death_acts, elites, sources = Counter(), Counter(), Counter(), Counter(), Counter()
    boss_hp, act2_hp, relics = defaultdict(list), [], defaultdict(Counter)
    skips = relic_total = unknown_sources = 0
    for run in runs:
        elite_count, first_act2 = 0, True
        for step in run.get("steps", []):
            if step["kind"] == "fight":
                key = (step.get("category", "unknown"), step.get("encounter", "unknown"))
                fights[key] += 1
                act = fight_act(step)
                if not step["won"]:
                    deaths[key] += 1
                    death_acts[act] += 1
                elite_count += key[0] == "elite"
                if "hp_before" in step:
                    if key[0] == "boss":
                        boss_hp[act].append(step["hp_before"])
                    if act == 2 and first_act2:
                        act2_hp.append(step["hp_before"])
                if act == 2:
                    first_act2 = False
            if step["kind"] not in ("pick", "decide"):
                continue
            source = step.get("source", "unknown")
            sources[source] += 1
            unknown_sources += source == "unknown"
            if step.get("decision") != "boss_relic":
                continue
            choice = step["choice"]
            if not 0 <= choice < len(step["options"]):
                continue
            option = step["options"][choice]
            relic = option.get("relic", "unknown") if isinstance(option, dict) else str(option)
            relic_total += 1
            skips += relic == "skip"
            relics[relic][source] += 1
        elites[elite_count] += 1
        # Non-combat deaths (e.g. events) have no lost fight step.
        if run["status"] == "died" and not any(s["kind"] == "fight" and not s["won"] for s in run.get("steps", [])):
            death_acts[run.get("act", 1 + max(0, run["floor"] - 1) // 17)] += 1
    lines = [f"## {label}", f"- n: {len(runs)}",
             f"- Acts cleared (acts: runs): {distribution(Counter(acts_cleared(r) for r in runs))}",
             f"- Floor mean ± SE: {mean_se([r['floor'] for r in runs])}",
             f"- Floor score mean ± SE: {mean_se([run_score(r, 'floors') for r in runs])}",
             f"- Deaths by act: {distribution(death_acts)}",
             f"- Elites/run (elites: runs): {distribution(elites)}",
             f"- Decision sources: {distribution(sources)}"]
    if unknown_sources:
        lines.append("- Unknown sources include annotations absent from canonical parquet run records.")
    for act, hp in sorted(boss_hp.items()):
        lines.append(f"- Act {act} boss entry HP mean ± SE: {mean_se(hp)}")
    lines.append(f"- Act 2 start HP (first fight) mean ± SE: {mean_se(act2_hp)}")
    lines += ["", "### Combat deaths (top 10)", "| Category | Encounter | Deaths | Fights | Death rate |",
              "|---|---|---:|---:|---:|"]
    for (category, encounter), n in deaths.most_common(10):
        total = fights[(category, encounter)]
        lines.append(f"| {category} | {encounter} | {n} | {total} | {n / total:.1%} |")
    lines += ["", "### Boss relic choices", "| Relic | Total | Net | Explore | Other/unknown |",
              "|---|---:|---:|---:|---:|"]
    for relic, counts in sorted(relics.items()):
        total = sum(counts.values())
        lines.append(f"| {relic} | {total} | {counts['net']} | {counts['explore']} | {total - counts['net'] - counts['explore']} |")
    rate = f"{skips / relic_total:.1%}" if relic_total else "n/a"
    lines.append(f"\nBoss relic skip rate: {rate} ({skips}/{relic_total}).")
    return "\n".join(lines)


def paired(first, second):
    def index(runs):
        by_seed = {r["seed"]: r for r in runs}
        if len(by_seed) != len(runs):
            raise ValueError("Paired comparison requires unique seeds in each input")
        return by_seed
    a, b = index(first), index(second)
    seeds = sorted(a.keys() & b.keys())
    lines = ["## Paired differences (DIR2 − DIR)", f"- Overlapping seeds: {len(seeds)}"]
    metrics = {"Floor score": lambda r: run_score(r, "floors"),
               "Act 1 clear": lambda r: int(acts_cleared(r) >= 1),
               "Act 2 clear": lambda r: int(acts_cleared(r) >= 2)}
    for name, metric in metrics.items():
        lines.append(f"- {name} mean difference ± SE: {mean_se([metric(b[s]) - metric(a[s]) for s in seeds])}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path)
    args = parser.parse_args()
    if not 1 <= len(args.dirs) <= 2:
        parser.error("provide DIR and optionally DIR2")
    data = [read_runs([d]) for d in args.dirs]
    print("\n\n".join(summary(r, str(d)) for d, r in zip(args.dirs, data)))
    if len(data) == 2:
        print("\n" + paired(*data))


if __name__ == "__main__":
    main()
