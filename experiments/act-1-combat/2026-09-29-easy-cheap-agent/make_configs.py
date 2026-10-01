#!/usr/bin/env python3
"""Writes every value_play config of the easy cheap-agent study into configs/ (one file per run; id = file stem).

    .venv/bin/python experiments/act-1-combat/2026-09-29-easy-cheap-agent/make_configs.py

Arms (fair; 8 particles unless stated; no random move; skip_diverged; search_salt = seed):
  S1 reference   mcts-20k-s0..s3
  S1 sweep       mcts-{10,25,50,100,200,500,1k,5k}-s0,s1
  S2 knobs       particles 1/2/4 and stop_factor 0.5 at KNEE and 500; gen1 value net at 50/200/1k;
                 t5 policy net (elite-trained priors) at 10/50/200
  S3 confirm     mcts-20k, mcts-500 and the pick(s) on confirm.csv (--confirm 100 [100:2 ...]; prefix easy-cheap-confirm)
"""
import argparse
from pathlib import Path

HERE = Path(__file__).parent
REL = "experiments/act-1-combat/2026-09-29-easy-cheap-agent"
GEN1 = "value_net_v1/2026-09-26/act1-gen1"
T5 = "value_net_v1/2026-09-29/bench-r2-t5"

p = argparse.ArgumentParser()
p.add_argument("--knee", type=int, default=None, help="S2: budget of the knee (lowest budget within 0.5 HP)")
p.add_argument("--confirm", nargs="*", default=(), help="S3: MCTS budgets (with optional particles, e.g. 100 or 100:2)")
a = p.parse_args()
out = HERE / "configs"
out.mkdir(exist_ok=True)


def write(stage, name, *, prefix="easy-cheap", leaf="mcts", sims, salt=0, particles=8, stop_factor=None, fights="fights.csv"):
    rid = f"{prefix}-{name}"
    net = {"gen1": GEN1, "t5": T5}.get(leaf, GEN1)
    lines = [f"# easy cheap-agent {stage}: {name}", "[run]", f'id = "{rid}"',
             f'input = "{net}"  # net arms: the net; mcts arms: leakage guard only', "workers = 11"]
    if leaf == "t5":
        lines += ['leaf = "policy_net"', "c_puct = 1.0", "fpu_reduction = 0.05", "prior_floor = 0.03"]
    elif leaf == "gen1":
        lines += ['leaf = "value_net"']
    else:
        lines += ['leaf = "guided_rollout"']
    lines += [f"simulations = {sims}", f"particles = {particles}", "random_move = false", "oracle = false"]
    if salt:
        lines += [f"search_salt = {salt}"]
    if stop_factor is not None:
        lines += [f"stop_factor = {stop_factor}"]
    lines += ["skip_diverged = true", 'query = """', "select * from combat_v3",
              "where id = 'act1-all-bosses-a20-scaled-search' and source_episode_id is null and category = 'easy'",
              f"  and episode_id in (select source_episode_id from read_csv_auto('{REL}/{fights}'))", '"""', ""]
    (out / f"{rid}.toml").write_text("\n".join(lines))
    return rid


def label(n):
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


SWEEP = (10, 25, 50, 100, 200, 500, 1000, 5000)
stages = {
    "S1ref": [write("S1", f"mcts-20k-s{s}", sims=20000, salt=s) for s in range(4)],
    "S1sweep": [write("S1", f"mcts-{label(b)}-s{s}", sims=b, salt=s) for b in SWEEP for s in (0, 1)],
    "S2nets": [write("S2", f"{leaf}-{label(b)}-s{s}", leaf=leaf, sims=b, salt=s)
               for leaf, budgets in (("gen1", (50, 200, 1000)), ("t5", (10, 50, 200))) for b in budgets for s in (0, 1)],
}
if a.knee:
    budgets = sorted({a.knee, 500})
    stages["S2knobs"] = [write("S2", f"mcts-{label(b)}-p{n}-s{s}", sims=b, salt=s, particles=n)
                         for b in budgets for n in (1, 2, 4) for s in (0, 1)] + \
                        [write("S2", f"mcts-{label(b)}-sf05-s{s}", sims=b, salt=s, stop_factor=0.5)
                         for b in budgets for s in (0, 1)]
if a.confirm:
    arms = [(20000, 8), (500, 8)] + [(int(c.split(":")[0]), int(c.split(":")[1]) if ":" in c else 8) for c in a.confirm]
    stages["S3confirm"] = [write("S3", f"mcts-{label(b)}{'' if n == 8 else f'-p{n}'}-s{s}", prefix="easy-cheap-confirm",
                                 sims=b, particles=n, salt=s, fights="confirm.csv")
                           for b, n in dict.fromkeys(arms) for s in (0, 1)]
for stage, ids in stages.items():
    print(stage, " ".join(ids))
