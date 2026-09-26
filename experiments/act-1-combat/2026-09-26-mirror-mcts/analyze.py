#!/usr/bin/env python3
"""Paired per-encounter comparison: candidate player vs teacher on the same dev fights, in HP-eq.

PYTHONPATH=python .venv/bin/python experiments/act-1-combat/2026-09-26-mirror-mcts/analyze.py \
    --candidate combat_v3/2026-09-26/act1-gen0-dev \
    --teacher combat_v3/2026-09-26/act1-teacher-dev combat_v3/2026-09-25/slime-v8-rollout-teacher-dev \
    [--value-run value_net_v1/2026-09-26/act1-gen0] [--json OUT] [--md OUT]

HP-eq per fight = (tv_candidate - tv_teacher) * (55 + max_hp)   (tv: win (35 + hp + 4*potions)/(55 + max_hp), loss 0)
CI: paired percentile bootstrap over fights (20,000 resamples, seed 0). McNemar: exact, fights won by only one side.
"Per act-1 run": per-encounter mean HP-eq weighted by that encounter's fights per train run (how often a run meets
it), CI by bootstrapping every encounter independently.
"""
import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

from sts_combat_rl.query import connect, sql_list

B = 20000


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def fights(db, run_ids):
    """episode_id -> (encounter, starting_hp, max_hp, tv, won, final_hp) of each fight's decision-0 row."""
    rows = db.sql(f"""select episode_id, category || '/' || encounter, starting_hp, starting_max_hp,
                             terminal_value, won, final_hp
                      from combat_v3 where run_id in {sql_list(run_ids)}
                        and row_kind = 'decision' and decision_index = 0""").fetchall()
    return {r[0]: r[1:] for r in rows}


def boot_mean(x, rng):
    return x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", nargs="+", required=True)
    p.add_argument("--teacher", nargs="+", required=True)
    p.add_argument("--value-run")
    p.add_argument("--json", type=Path)
    p.add_argument("--md", type=Path)
    a = p.parse_args()
    db = connect()
    db.sql("set enable_progress_bar = false")
    cand, teach = fights(db, a.candidate), fights(db, a.teacher)
    common = sorted(set(cand) & set(teach))
    if len(common) < len(cand):
        print(f"warning: {len(cand) - len(common)} candidate fights have no teacher fight; dropped")
    assert all(cand[e][:2] == teach[e][:2] for e in common), "unpaired fights (encounter / starting HP differ)"
    enc = np.array([cand[e][0] for e in common])
    col = lambda d, i: np.array([d[e][i] for e in common], dtype=float)
    scale = 55 + col(cand, 2)
    hp_eq = (col(cand, 3) - col(teach, 3)) * scale
    wc, wt = col(cand, 4).astype(bool), col(teach, 4).astype(bool)
    hl_c, hl_t = col(cand, 1) - col(cand, 5), col(teach, 1) - col(teach, 5)  # HP lost (start - end)

    # encounter frequency per act-1 run (train split, bootstrap runs)
    freq = dict(db.sql("""select category || '/' || encounter, count(distinct episode_id) / (
                              select count(distinct run_seed) from combat_v3 where id like 'act1-a20%'
                                and source_episode_id is null and not (id = 'act1-a20-8' and run_seed % 10 < 4))
                          from combat_v3 where id like 'act1-a20%' and source_episode_id is null
                            and not (id = 'act1-a20-8' and run_seed % 10 < 4) group by 1""").fetchall())
    val = {}
    if a.value_run:
        s, d, i = a.value_run.split("/")
        meta = json.loads(Path(f"runs/schema={s}/date={d}/id={i}/out/value_checkpoint.json").read_text())
        val = meta["metrics"].get("validation_by_encounter", {})

    rng = np.random.default_rng(0)
    rows, per_run_boot, per_run = [], np.zeros(B), 0.0
    for name in sorted(set(enc)):
        m = enc == name
        x = hp_eq[m]
        bm = boot_mean(x, rng)
        both = wc[m] & wt[m]
        only_c, only_t = int((wc[m] & ~wt[m]).sum()), int((wt[m] & ~wc[m]).sum())
        v = val.get(name)
        r = {"encounter": name, "fights": int(m.sum()), "hp_eq": float(x.mean()),
             "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
             "win_teacher": float(wt[m].mean()), "win_candidate": float(wc[m].mean()),
             "only_candidate": only_c, "only_teacher": only_t, "mcnemar_p": mcnemar(only_c, only_t),
             "hp_lost_teacher_both_won": float(hl_t[m][both].mean()), "hp_lost_candidate_both_won": float(hl_c[m][both].mean()),
             "per_run_freq": freq.get(name, 0.0),
             "val_r2": (1 - v["mse"] / v["baseline_mse"]) if v else None, "val_mse": v["mse"] if v else None}
        rows.append(r)
        per_run += r["per_run_freq"] * r["hp_eq"]
        per_run_boot += r["per_run_freq"] * bm
    allx = hp_eq
    ball = boot_mean(allx, rng)
    total = {"fights": len(allx), "hp_eq_mean_per_fight": float(allx.mean()),
             "ci95": [float(np.percentile(ball, 2.5)), float(np.percentile(ball, 97.5))],
             "per_act1_run": float(per_run),
             "per_act1_run_ci95": [float(np.percentile(per_run_boot, 2.5)), float(np.percentile(per_run_boot, 97.5))],
             "win_teacher": float(wt.mean()), "win_candidate": float(wc.mean())}
    result = {"candidate": a.candidate, "teacher": a.teacher, "value_run": a.value_run, "total": total,
              "encounters": rows}

    lines = ["| encounter | fights | Δ HP-eq / fight (95% CI) | win % teacher → net | only net / only teacher (McNemar p) "
             "| HP lost, both won: teacher → net | fights per run | val R² |",
             "|---|---:|---|---|---|---|---:|---:|"]
    for r in rows:
        r2 = "" if r["val_r2"] is None else f"{r['val_r2']:.3f}"
        lines.append(f"| {r['encounter']} | {r['fights']} | {r['hp_eq']:+.1f} ({r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}) "
                     f"| {100 * r['win_teacher']:.1f} → {100 * r['win_candidate']:.1f} "
                     f"| {r['only_candidate']} / {r['only_teacher']} ({r['mcnemar_p']:.3f}) "
                     f"| {r['hp_lost_teacher_both_won']:.1f} → {r['hp_lost_candidate_both_won']:.1f} "
                     f"| {r['per_run_freq']:.2f} | {r2} |")
    t = total
    lines += ["", f"All {t['fights']} fights: {t['hp_eq_mean_per_fight']:+.2f} HP-eq/fight "
                  f"({t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}); wins {100 * t['win_teacher']:.1f}% → {100 * t['win_candidate']:.1f}%.",
              f"Per act-1 run (encounters weighted by how often a run meets them): {t['per_act1_run']:+.1f} HP-eq "
              f"({t['per_act1_run_ci95'][0]:+.1f}, {t['per_act1_run_ci95'][1]:+.1f})."]
    text = "\n".join(lines)
    print(text)
    if a.json:
        a.json.write_text(json.dumps(result, indent=2) + "\n")
    if a.md:
        a.md.write_text(text + "\n")


if __name__ == "__main__":
    main()
