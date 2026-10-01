"""Shared loader for the elite-bench deck: one row per (run, fight) from the runs' parquet (not the view).

Run ids are PREFIX-ARM-sSALT (fair arms) or PREFIX-oracle-BUDGET. ARM = agent-budget[-pN]. HP-eq of a fight =
terminal_value * (55 + max HP): 0 on a death, else 35 + final HP + 4 * potions kept.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ELITES = ("gremlin_nob", "lagavulin", "three_sentries")
ID_RE = re.compile(r"^(?P<prefix>[a-z0-9]+)-(?P<arm>.+?)(?:-s(?P<salt>\d+))?$")


def load(prefix: str, store: str = "runs") -> dict:
    """{run_id: {episode_id: fight dict}} for every finished-or-not run PREFIX-* under store/schema=combat_v3."""
    glob = str(ROOT / store / "schema=combat_v3" / "date=*" / f"id={prefix}-*" / "out" / "*.parquet")
    rows = duckdb.sql(f"""
        select id, episode_id, any_value(run_seed), any_value(encounter), any_value(starting_hp),
               any_value(starting_max_hp), any_value(won), any_value(terminal_value), max(turn), count(*)
        from read_parquet('{glob}', hive_partitioning = true, union_by_name = true)
        where row_kind = 'decision'
        group by id, episode_id""").fetchall()
    runs: dict = defaultdict(dict)
    for rid, ep, seed, enc, hp, mhp, won, tv, turns, decisions in rows:
        runs[rid][ep] = dict(seed=seed, enc=enc, hp=hp, mhp=mhp, won=bool(won), hpeq=float(tv) * (55 + mhp),
                             turns=int(turns), decisions=int(decisions))
    return dict(runs)


def parse(run_id: str) -> tuple[str, int]:
    m = ID_RE.match(run_id)
    return m["arm"], int(m["salt"] or 0)


def arms(runs: dict) -> dict:
    """{arm: {salt: {episode: fight}}}."""
    out: dict = defaultdict(dict)
    for rid, fights in runs.items():
        arm, salt = parse(rid)
        out[arm][salt] = fights
    return dict(out)


def cluster_ci(values, seeds, rng, n=10000):
    """Mean and 95% percentile CI, bootstrapping source run seeds (fights of one run share a deck)."""
    values = np.asarray(values, float)
    groups = defaultdict(list)
    for v, s in zip(values, seeds):
        groups[s].append(v)
    sums = np.array([sum(g) for g in groups.values()])
    counts = np.array([len(g) for g in groups.values()])
    if len(sums) < 2:
        return float(values.mean()) if len(values) else float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(sums), (n, len(sums)))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def fmt(ci) -> str:
    m, lo, hi = ci
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
