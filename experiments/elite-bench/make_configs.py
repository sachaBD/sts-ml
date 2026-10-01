#!/usr/bin/env python3
"""Writes every value_play config of the elite-bench deck into configs/ (one file per run; id = file stem).

    .venv/bin/python experiments/elite-bench/make_configs.py [--prefix bench] [--fights FILE] [--pivotal FILE]

Arms (all fair unless oracle; 8 particles unless stated; no random move; skip_diverged):
  S1 oracle-20k (all fights), oracle-50k (subset)
  S2 mcts-20k-s0..s3            S3 v3-20k-s0..s3 (elite-v3-t2, policy priors, c_puct 1.0)
  S4 mcts-{1k,2k,10k,50k}-s0,s1
  S5 mcts-20k-s4..s7, v3-20k-s4..s7 on pivotal fights (list written by pivotal.py after S1-S3)
  S6 mcts-5k / v3-5k / v3-50k -s0,s1 on the subset
  S7 mcts-20k-p32-s0..s3 on pivotal fights
"""
import argparse
from pathlib import Path

HERE = Path(__file__).parent
REL = "experiments/elite-bench"
NET = "value_net_v1/2026-09-27/elite-v3-t2"

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="bench")
p.add_argument("--fights", default=f"{REL}/bench_fights.csv")
p.add_argument("--subset", default=f"{REL}/bench_subset.csv")
p.add_argument("--pivotal", default=f"{REL}/pivotal_fights.csv")
p.add_argument("--out", default=str(HERE / "configs"))
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)


def write(stage, name, fights, *, leaf="mcts", sims, salt=0, oracle=False, particles=8):
    rid = f"{a.prefix}-{name}"
    lines = [f"# elite-bench {stage}: {name}", "[run]", f'id = "{rid}"', f'input = "{NET}"  # v3 arms: the net; others: leakage guard only',
             "workers = 11"]
    if leaf == "v3":
        lines += ['leaf = "policy_net"', "c_puct = 1.0", "fpu_reduction = 0.05", "prior_floor = 0.03"]
    else:
        lines += ['leaf = "guided_rollout"']
    lines += [f"simulations = {sims}"]
    if not oracle:
        lines += [f"particles = {particles}"]
    lines += ["random_move = false", f"oracle = {'true' if oracle else 'false'}"]
    if salt:
        lines += [f"search_salt = {salt}"]
    lines += ["skip_diverged = true", 'query = """', "select * from combat_v3",
              "where id = 'act1-all-bosses-a20-scaled-search' and source_episode_id is null and category = 'elite'",
              "  and run_seed % 10 in (0, 1)",
              f"  and episode_id in (select source_episode_id from read_csv_auto('{fights}'))", '"""', ""]
    (out / f"{rid}.toml").write_text("\n".join(lines))
    return rid


K = {"1k": 1000, "2k": 2000, "5k": 5000, "10k": 10000, "20k": 20000, "50k": 50000}
stages = {
    "S1": [write("S1", "oracle-20k", a.fights, sims=K["20k"], oracle=True),
           write("S1", "oracle-50k", a.subset, sims=K["50k"], oracle=True)],
    "S2": [write("S2", f"mcts-20k-s{s}", a.fights, sims=K["20k"], salt=s) for s in range(4)],
    "S3": [write("S3", f"v3-20k-s{s}", a.fights, leaf="v3", sims=K["20k"], salt=s) for s in range(4)],
    "S4": [write("S4", f"mcts-{b}-s{s}", a.fights, sims=K[b], salt=s) for b in ("1k", "2k", "10k", "50k") for s in (0, 1)],
    "S5": [write("S5", f"{agent}-20k-s{s}", a.pivotal, leaf=agent, sims=K["20k"], salt=s)
           for s in range(4, 8) for agent in ("mcts", "v3")],
    "S6": [write("S6", f"{agent}-{b}-s{s}", a.subset, leaf=agent, sims=K[b], salt=s)
           for agent, b in (("mcts", "5k"), ("v3", "5k"), ("v3", "50k")) for s in (0, 1)],
    "S7": [write("S7", f"mcts-20k-p32-s{s}", a.pivotal, sims=K["20k"], salt=s, particles=32) for s in range(4)],
}
for stage, ids in stages.items():
    print(stage, " ".join(ids))
