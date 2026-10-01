#!/usr/bin/env python3
"""Writes the hard-budget study's value_play configs into configs/ (one file per run; id = file stem).

    .venv/bin/python experiments/act-1-combat/2026-09-29-hard-budget/make_configs.py

Arms: MCTS (guided rollout, 8 particles, fair, no random move, skip_diverged) at 500 / 1k / 2k / 5k / 10k simulations,
search_salt 0 and 1, on fights.csv. Plus pilot-mcts-10k-s0: 20 fights (2 per encounter) to time the study.
"""
from pathlib import Path

HERE = Path(__file__).parent
REL = "experiments/act-1-combat/2026-09-29-hard-budget"
GUARD = "value_net_v1/2026-09-26/act1-gen1"  # leakage guard only (no net is used)
out = HERE / "configs"
out.mkdir(exist_ok=True)


def write(name, sims, salt, where=""):
    rid = f"hard-budget-{name}"
    lines = [f"# hard-budget: {name}", "[run]", f'id = "{rid}"', f'input = "{GUARD}"  # leakage guard only', "workers = 11",
             'leaf = "guided_rollout"', f"simulations = {sims}", "particles = 8", "random_move = false", "oracle = false"]
    if salt:
        lines += [f"search_salt = {salt}"]
    lines += ["skip_diverged = true", 'query = """', "select * from combat_v3",
              "where id = 'act1-all-bosses-a20-scaled-search' and source_episode_id is null and category = 'hard'",
              f"  and episode_id in (select source_episode_id from read_csv_auto('{REL}/fights.csv'){where})", '"""', ""]
    (out / f"{rid}.toml").write_text("\n".join(lines))
    return rid


label = lambda n: f"{n // 1000}k" if n >= 1000 else str(n)
print("pilot", write("pilot-mcts-10k-s0", 10000, 0,
                     " qualify row_number() over (partition by encounter order by source_episode_id) <= 2"))
print("main", " ".join(write(f"mcts-{label(b)}-s{s}", b, s) for b in (10000, 5000, 2000, 1000, 500) for s in (0, 1)))
