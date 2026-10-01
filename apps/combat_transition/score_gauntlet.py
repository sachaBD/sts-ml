#!/usr/bin/env python3
"""Score a gauntlet_v1 run's fights with a frozen combat_outcome_v1 model and compare options with the real agent.

  PYTHONPATH=. .venv/bin/python apps/combat_transition/score_gauntlet.py GAUNTLET_RUN MODEL_RUN [--out DIR]

Every gauntlet row carries the pre-init state (`pre`) of that option's fight, so the model sees exactly the inputs the
real agent played from (same reward-state HP / relics / potions, same augmentation picks). Per fight the model gives
p = P(win) and E[HP | win] (bin centres), so
    E[final HP] = p * E[HP | win]          (death = 0 HP)
    E[score]    = start - E[final HP] + death_penalty * (1 - p)     (the gauntlet's score, lower is better)
Per reward state, option and fight kind (elite / boss), fights are averaged over the shared seeds. Each non-baseline
option is paired with the SimpleAgent baseline option of the same reward state (same seeds): real vs predicted
differences in win rate, unconditional final HP and score, against a card-invariant reference that predicts delta 0.
Cluster bootstrap intervals resample source run seeds (reward-state lineage). Descriptive pilot.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs.run import RUNS, SCRATCH  # noqa: E402
from models.combat_outcome.combat_outcome_v1 import CombatOutcomeV1  # noqa: E402
from models.combat_outcome.learn import batch  # noqa: E402

METRICS = ("win", "final_hp", "score")


def out_dir(run_id):
    schema, date, rid = run_id.split("/")
    for root in (RUNS, SCRATCH):
        path = root / f"schema={schema}" / f"date={date}" / f"id={rid}" / "out"
        if path.exists():
            return path
    raise FileNotFoundError(run_id)


def read_rows(path):
    return [r for p in sorted(path.glob("*.parquet")) for r in pq.read_table(p).to_pylist()]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else None


def pair_stats(pairs, seed=0, draws=2000):
    """pairs: dicts with run_seed and real_/pred_ deltas. Correlation, MAE of the model vs the delta-0 reference, sign
    agreement; 95% bootstrap over run seeds for the MAE difference (model - reference)."""
    out = {"pairs": len(pairs), "run_seeds": len({p["run_seed"] for p in pairs})}
    by_seed = defaultdict(list)
    for p in pairs:
        by_seed[p["run_seed"]].append(p)
    seeds = sorted(by_seed)
    rng = random.Random(seed)
    for m in METRICS:
        real, pred = [p[f"real_{m}"] for p in pairs], [p[f"pred_{m}"] for p in pairs]
        mae = mean([abs(r - q) for r, q in zip(real, pred)])
        ref = mean([abs(r) for r in real])
        nz = [(r, q) for r, q in zip(real, pred) if r != 0]
        boot = []
        for _ in range(draws):
            sample = [p for s in (rng.choice(seeds) for _ in seeds) for p in by_seed[s]]
            boot.append(mean([abs(p[f"real_{m}"] - p[f"pred_{m}"]) - abs(p[f"real_{m}"]) for p in sample]))
        boot.sort()
        out[m] = {"corr": corr(real, pred), "mae_model": mae, "mae_delta0": ref,
                  "mae_model_minus_delta0": mae - ref,
                  "ci95_model_minus_delta0": [boot[int(0.025 * draws)], boot[int(0.975 * draws) - 1]],
                  "sign_agreement_nonzero": mean([(r > 0) == (q > 0) for r, q in nz]),
                  "mean_real_delta": mean(real), "mean_pred_delta": mean(pred),
                  "sd_real_delta": (mean([(r - mean(real)) ** 2 for r in real]) or 0) ** 0.5}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gauntlet_run")
    ap.add_argument("model_run")
    ap.add_argument("--transition_run", default="combat_transition_v1/2026-09-29/act1-eval-mcts-a20-checked",
                    help="training cohort, for the train-distribution comparison of deck size / floor")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    gdir = out_dir(args.gauntlet_run)
    summary = json.loads((gdir / "summary.json").read_text())
    penalty = summary["death_penalty"]
    rows = read_rows(gdir)
    ck = torch.load(out_dir(args.model_run) / "model.pt", weights_only=False)
    model = CombatOutcomeV1(**ck["args"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    centers = torch.tensor(ck["centers"])
    for r in rows:
        r["pre"] = json.loads(r["pre"])
    with torch.no_grad():
        for i in range(0, len(rows), 1024):
            chunk = rows[i:i + 1024]
            b, _, _ = batch(chunk, ck["encounters"], "cpu")
            wl, hl = model(b)
            p, hp_win = torch.sigmoid(wl), (torch.softmax(hl, -1) * centers).sum(-1)
            for r, pi, hi in zip(chunk, p.tolist(), hp_win.tolist()):
                start = r["start_hp"]
                r["pred_win"], r["pred_hp_if_win"] = pi, hi
                r["pred_final_hp"] = pi * hi
                r["pred_score"] = start - pi * hi + penalty * (1 - pi)
                r["real_win"], r["real_final_hp"] = float(r["won"]), float(r["final_hp"])  # final_hp = 0 on death
                r["real_score"] = r["score"]
                assert abs(r["real_score"] - (start - r["final_hp"] + penalty * (not r["won"]))) < 1e-9

    # per (episode, option, kind): means over the shared seeds
    cell = defaultdict(list)
    for r in rows:
        cell[r["episode_id"], r["option"], r["kind"]].append(r)
    agg = {k: {f"{s}_{m}": mean([f[f"{s}_{m}"] for f in fs]) for s in ("real", "pred") for m in METRICS}
           for k, fs in cell.items()}
    info = {}
    for (e, o, kind), fs in cell.items():
        f = fs[0]
        info.setdefault(e, {"run_seed": f["run_seed"], "floor": f["floor"], "options": {}, "baseline": None})
        info[e]["options"][o] = f["card"]
        if f["baseline"]:
            info[e]["baseline"] = o
    pairs = defaultdict(list)  # kind -> pairs
    for (e, o, kind), a in agg.items():
        base = info[e]["baseline"]
        if o == base:
            continue
        b = agg[e, base, kind]
        pairs[kind].append({"episode_id": e, "run_seed": info[e]["run_seed"], "option": o,
                            **{f"{s}_{m}": a[f"{s}_{m}"] - b[f"{s}_{m}"] for s in ("real", "pred") for m in METRICS}})

    # calibration of levels per kind / encounter (fights, not paired)
    levels = {}
    for key in sorted({(r["kind"], r["encounter"]) for r in rows}):
        fs = [r for r in rows if (r["kind"], r["encounter"]) == key]
        levels["/".join(key)] = {"fights": len(fs), **{f"{s}_{m}": mean([f[f"{s}_{m}"] for f in fs])
                                                         for s in ("real", "pred") for m in METRICS}}

    # distribution shift: deck size / floor / picks of gauntlet fights vs training fights of the same kind
    tdir = out_dir(args.transition_run)
    train = [t for t in read_rows(tdir) if t["replay"] == "ok" and t["bucket"] in (4, 5, 6, 7, 8)]
    kind_of = {"boss": "boss", "elite": "elite"}
    shift = {}
    for kind in ("elite", "boss"):
        g = [r for r in rows if r["kind"] == kind]
        t = [x for x in train if x["category"] == kind_of[kind]]
        dist = lambda xs: {"mean": mean(xs), "min": min(xs), "max": max(xs)} if xs else None
        shift[kind] = {"gauntlet_deck": dist([len(r["pre"]["deck"]) for r in g]),
                       "train_deck": dist([len(x["pre"]["deck"]) for x in t]),
                       "gauntlet_reward_floor": dist([r["floor"] for r in g]),
                       "train_floor": dist([x["floor"] for x in t]),
                       "gauntlet_picks": dist([r["picks"] for r in g]),
                       "gauntlet_start_hp": dist([r["start_hp"] for r in g]),
                       "train_start_hp": dist([x["pre"]["hp"] for x in t]),
                       "gauntlet_relics": dist([len(r["pre"]["relics"]) for r in g]),
                       "train_relics": dist([len(x["pre"]["relics"]) for x in t])}

    report = {"gauntlet_run": args.gauntlet_run, "model_run": args.model_run, "reward_states": len(info),
              "fights": len(rows), "death_penalty": penalty,
              "unseen_encounters": sorted({r["encounter"] for r in rows} - set(ck["encounters"])),
              "score_formula": "start - p*E[HP|win] + penalty*(1-p); real: start - final_hp(0 on death) + penalty*died",
              "paired_vs_baseline": {k: pair_stats(v) for k, v in sorted(pairs.items())},
              "levels_by_encounter": levels, "distribution_shift": shift,
              "states": [{"episode_id": e, **v, "options": {str(o): c for o, c in v["options"].items()}}
                         for e, v in sorted(info.items())],
              "note": "descriptive pilot; deltas pair options of one reward state on shared seeds; CIs resample run "
                      "seeds and ignore model training / selection uncertainty"}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "score_gauntlet.json").write_text(text)
        (args.out / "pairs.json").write_text(json.dumps(pairs, indent=1))
    print(json.dumps({k: report[k] for k in ("reward_states", "fights", "unseen_encounters", "paired_vs_baseline",
                                             "levels_by_encounter", "distribution_shift")}, indent=1))


if __name__ == "__main__":
    main()
