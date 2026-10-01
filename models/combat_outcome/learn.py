#!/usr/bin/env python3
"""Train and evaluate the pre-combat outcome model (topology combat_outcome_v1) on combat_transition_v1 rows.

  PYTHONPATH=. .venv/bin/python -m runs.run combat_outcome_v1 ID --input TRANSITION_RUN -- \
      .venv/bin/python models/combat_outcome/learn.py TRANSITION_RUN --out {out}

Rows: replay = 'ok', no random move, not oracle. Split by run_seed % 10: dev = buckets 0-1 (reported), buckets 2-3
never loaded (reserved confirmation), early stopping = bucket 9, training = buckets 4-8.
Endpoint: in-battle HP before exitBattle (combat_v3 final_hp), conditional on a win.
Baseline (same endpoint and survivor conditioning): per encounter, logistic P(win) on start HP and a multinomial HP
bin distribution on start HP (global terms + L2-shrunk per-encounter terms).
out/: model.pt, report.json, report.md.
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runs.run import RUNS  # noqa: E402
from models.combat_outcome.combat_outcome_v1 import BINS, HP_BIN, HP_CAP, CombatOutcomeV1, hp_bin  # noqa: E402

DEV, RESERVED, EARLY_STOP = {0, 1}, {2, 3}, {9}
torch.manual_seed(0)


def load(run_id):
    schema, date, rid = run_id.split("/")
    path = RUNS / f"schema={schema}" / f"date={date}" / f"id={rid}" / "out"
    rows = [r for p in sorted(path.glob("*.parquet")) for r in pq.read_table(p).to_pylist()]
    counts = defaultdict(int)
    kept = []
    for r in rows:
        if r["bucket"] in RESERVED:
            counts["reserved_buckets_2_3"] += 1
        elif r["replay"] != "ok":
            counts[f"replay_{r['replay']}"] += 1
        elif r["any_random"] or r["oracle"]:
            counts["random_or_oracle"] += 1
        else:
            kept.append(r)
    return kept, dict(counts)


def batch(rows, encounters, device):
    B = len(rows)
    C = max(len(r["pre"]["deck"]) for r in rows)
    R = max(max(len(r["pre"]["relics"]) for r in rows), 1)
    P = max(max(len(r["pre"]["potions"]) for r in rows), 1)
    t = lambda *s: torch.zeros(*s)
    b = {"card_id": t(B, C).long(), "card_up": t(B, C), "card_misc": t(B, C), "card_mask": t(B, C),
         "relic_id": t(B, R).long(), "relic_data": t(B, R), "relic_mask": t(B, R),
         "potion_id": t(B, P).long(), "potion_mask": t(B, P), "encounter": t(B).long(), "scalars": t(B, 5)}
    for i, r in enumerate(rows):
        pre = r["pre"]
        for j, c in enumerate(pre["deck"]):
            b["card_id"][i, j], b["card_up"][i, j], b["card_misc"][i, j], b["card_mask"][i, j] = \
                c["card_id"] + 1, c["upgraded"], c["misc"], 1
        for j, x in enumerate(pre["relics"]):
            b["relic_id"][i, j], b["relic_data"][i, j], b["relic_mask"][i, j] = x["relic_id"] + 1, x["data"], 1
        for j, x in enumerate(pre["potions"]):
            b["potion_id"][i, j], b["potion_mask"][i, j] = x["potion_id"] + 1, 1
        b["encounter"][i] = encounters.get(r["encounter"], 0)
        b["scalars"][i] = torch.tensor([pre["hp"] / 100, pre["max_hp"] / 100, pre["hp"] / pre["max_hp"],
                                        pre["potion_capacity"] / 5, math.log1p(len(pre["deck"]))])
    b = {k: v.to(device) for k, v in b.items()}
    won = torch.tensor([float(r["won"]) for r in rows], device=device)
    hp = torch.tensor([r["final_hp"] for r in rows], device=device)
    return b, won, hp


def losses(win_logit, hp_logits, won, hp):
    bce = F.binary_cross_entropy_with_logits(win_logit, won)
    w = won > 0
    nll = F.cross_entropy(hp_logits[w], hp_bin(hp[w])) if w.any() else win_logit.sum() * 0
    return bce, nll


class Baseline(torch.nn.Module):
    """Per-encounter P(win) and HP-bin logits linear in start HP: global + per-encounter terms."""

    def __init__(self, encounters):
        super().__init__()
        n = encounters + 1
        self.g = torch.nn.Parameter(torch.zeros(2, 1 + BINS))  # (intercept, slope) x (win, bins)
        self.e = torch.nn.Parameter(torch.zeros(n, 2, 1 + BINS))

    def forward(self, b):
        x = b["scalars"][:, 0]  # start hp / 100
        coef = self.g[None] + self.e[b["encounter"]]
        out = coef[:, 0] + coef[:, 1] * x[:, None]
        return out[:, 0], out[:, 1:]

    def penalty(self):
        return 1e-2 * (self.e ** 2).sum()


def fit(model, train, stop, epochs, lr, wd, batch_size=256, penalty=False, check_start=False):
    # the per-encounter linear terms (baseline form) are not weight-decayed
    linear = [p for n, p in model.named_parameters() if n == "enc_linear"]
    rest = [p for n, p in model.named_parameters() if n != "enc_linear"]
    opt = torch.optim.AdamW([{"params": rest}, {"params": linear, "weight_decay": 0.0}], lr=lr, weight_decay=wd)
    best, best_state, bad, best_epoch = float("inf"), None, 0, 0
    n = len(train[1])
    if check_start:  # epoch 0 = the initial (baseline-equivalent) model may be the best
        model.eval()
        with torch.no_grad():
            bce, nll = losses(*model(stop[0]), stop[1], stop[2])
        best, best_state = (bce + nll * stop[1].mean()).item(), {k: x.clone() for k, x in model.state_dict().items()}
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(n).split(batch_size):
            b = {k: v[idx] for k, v in train[0].items()}
            bce, nll = losses(*model(b), train[1][idx], train[2][idx])
            loss = bce + nll + (model.penalty() / n * len(idx) if penalty else 0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            bce, nll = losses(*model(stop[0]), stop[1], stop[2])
            v = (bce + nll * stop[1].mean()).item()  # per-fight joint NLL
        if v < best - 1e-4:
            best, best_state, bad, best_epoch = v, {k: x.clone() for k, x in model.state_dict().items()}, 0, epoch + 1
        else:
            bad += 1
            if bad >= 20:
                break
    model.load_state_dict(best_state)
    return best_epoch, best


def predict(model, data):
    model.eval()
    with torch.no_grad():
        wl, hl = model(data[0])
    return torch.sigmoid(wl), torch.softmax(hl, -1)


def metrics(p, probs, won, hp, centers):
    eps = 1e-6
    out = {"fights": len(won), "deaths": int((won == 0).sum()),
           "brier": ((p - won) ** 2).mean().item(),
           "log_loss": -(won * torch.log(p + eps) + (1 - won) * torch.log(1 - p + eps)).mean().item(),
           "mean_p_win": p.mean().item(), "observed_win": won.mean().item()}
    w = won > 0
    if w.any():
        pw, h = probs[w], hp[w]
        mean = (pw * centers).sum(-1)
        cdf = pw.cumsum(-1)
        lo = (cdf < 0.1).sum(-1)
        hi = (cdf < 0.9).sum(-1)
        bins = hp_bin(h)
        out.update({"wins": int(w.sum()), "hp_mae": (mean - h).abs().mean().item(),
                    "hp_bin_nll": -torch.log(pw.gather(1, bins[:, None]).squeeze(1) + eps).mean().item(),
                    "hp_80_interval_coverage": ((bins >= lo) & (bins <= hi)).float().mean().item()})
    return out


def reliability(p, won, k=10):
    rows = []
    for i in range(k):
        m = (p >= i / k) & ((p < (i + 1) / k) if i < k - 1 else (p <= 1))
        if m.any():
            rows.append({"bin": f"{i / k:.1f}-{(i + 1) / k:.1f}", "n": int(m.sum()),
                         "pred": p[m].mean().item(), "obs": won[m].mean().item()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transition_run")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = "cpu"
    rows, excluded = load(args.transition_run)
    split = lambda s: [r for r in rows if r["bucket"] in s]
    dev, stop = split(DEV), split(EARLY_STOP)
    train = [r for r in rows if r["bucket"] not in DEV | EARLY_STOP]
    names = sorted({r["encounter"] for r in train})
    encounters = {e: i + 1 for i, e in enumerate(names)}  # 0 = unseen
    overflow = sum(r["won"] and r["final_hp"] > HP_CAP for r in rows)
    over_hp = [r["final_hp"] for r in train if r["won"] and r["final_hp"] > HP_CAP]
    centers = torch.tensor([(i * HP_BIN + 1 + i * HP_BIN + HP_BIN) / 2 for i in range(BINS - 1)]
                           + [sum(over_hp) / len(over_hp) if over_hp else HP_CAP + HP_BIN / 2 + 0.5])
    data = {k: batch(v, encounters, device) for k, v in (("train", train), ("stop", stop), ("dev", dev))}

    results, preds = {}, {}
    baseline = Baseline(len(names))
    model = CombatOutcomeV1(encounters=len(names) + 1, width=16, hidden=32, dropout=0.3)
    for name, net, kw in (("baseline", baseline, dict(lr=0.05, wd=0.0, penalty=True)),
                          ("model", model, dict(lr=3e-3, wd=1e-1))):
        t0 = time.time()
        if name == "model":  # residual on the fitted baseline: start from its per-encounter linear terms
            with torch.no_grad():
                model.enc_linear.copy_(baseline.g[None] + baseline.e)
                model.win_out.weight.zero_(), model.win_out.bias.zero_()
                model.hp_out.weight.zero_(), model.hp_out.bias.zero_()
        epochs, stop_loss = fit(net, data["train"], data["stop"], 500, check_start=name == "model", **kw)
        p, probs = predict(net, data["dev"])
        won, hp = data["dev"][1], data["dev"][2]
        per = {}
        for key in sorted({(r["category"], r["encounter"]) for r in dev}):
            m = torch.tensor([(r["category"], r["encounter"]) == key for r in dev])
            per[f"{key[0]}/{key[1]}"] = metrics(p[m], probs[m], won[m], hp[m], centers)
        for cat in sorted({r["category"] for r in dev}):
            m = torch.tensor([r["category"] == cat for r in dev])
            per[f"{cat}/ALL"] = metrics(p[m], probs[m], won[m], hp[m], centers)
        preds[name] = (p, probs)
        results[name] = {"best_epoch": epochs, "early_stop_loss": stop_loss, "train_seconds": time.time() - t0,
                         "overall": metrics(p, probs, won, hp, centers), "reliability": reliability(p, won),
                         "by_group": per}
        if name == "model":
            torch.save({"kind": model.KIND, "args": model.args, "state_dict": model.state_dict(),
                        "encounters": encounters, "centers": centers.tolist()}, args.out / "model.pt")
            b = {k: v[:1024] for k, v in data["dev"][0].items()}
            with torch.no_grad():
                model(b)
                t0 = time.time()
                for _ in range(20):
                    model(b)
            n = len(b["encounter"])
            results[name]["throughput_cpu_per_s"] = 20 * n / (time.time() - t0)
            results[name]["throughput_batch"] = n
            results[name]["params"] = sum(x.numel() for x in model.parameters())

    # per dev fight: model - baseline losses; bootstrap clustered by run seed (1000 resamples)
    won, hp = data["dev"][1], data["dev"][2]
    eps = 1e-6
    def per_fight(p, probs):
        mean = (probs * centers).sum(-1)
        nll = -torch.log(probs.gather(1, hp_bin(hp)[:, None]).squeeze(1) + eps)
        return {"brier": (p - won) ** 2,
                "log_loss": -(won * torch.log(p + eps) + (1 - won) * torch.log(1 - p + eps)),
                "hp_abs_err": torch.where(won > 0, (mean - hp).abs(), torch.nan),
                "hp_bin_nll": torch.where(won > 0, nll, torch.nan)}
    fb, fm = per_fight(*preds["baseline"]), per_fight(*preds["model"])
    seeds = sorted({r["run_seed"] for r in dev})
    idx_by_seed = defaultdict(list)
    for i, r in enumerate(dev):
        idx_by_seed[r["run_seed"]].append(i)
    gen = torch.Generator().manual_seed(0)
    boot = {}
    for group in ["ALL", *sorted({r["category"] for r in dev})]:
        mask = torch.tensor([group == "ALL" or r["category"] == group for r in dev])
        boot[group] = {}
        for k in fb:
            d = (fm[k] - fb[k])
            draws = []
            for _ in range(1000):
                pick = torch.randint(len(seeds), (len(seeds),), generator=gen)
                ii = torch.tensor([i for j in pick.tolist() for i in idx_by_seed[seeds[j]]])
                x = d[ii][mask[ii]]
                x = x[~torch.isnan(x)]
                if len(x):
                    draws.append(x.mean().item())
            full = d[mask]
            full = full[~torch.isnan(full)]
            if len(full) and draws:
                q = torch.tensor(draws).quantile(torch.tensor([0.025, 0.975])).tolist()
                boot[group][k] = {"delta": full.mean().item(), "ci95": q, "n": len(full)}
    results["model_minus_baseline_bootstrap"] = boot

    count = lambda rs: {"fights": len(rs), "runs": len({r["run_seed"] for r in rs}),
                        "deaths": sum(not r["won"] for r in rs),
                        "deaths_by_category": {c: sum(not r["won"] for r in rs if r["category"] == c)
                                               for c in sorted({r["category"] for r in rs})}}
    report = {"transition_run": args.transition_run, "excluded": excluded, "overflow_wins_gt_100": overflow,
              "overflow_bin_center": centers[-1].item(), "hp_bins": f"{HP_BIN}-HP bins 1..{HP_CAP}, overflow > {HP_CAP}",
              "endpoint": "in-battle HP before exitBattle (combat_v3 final_hp), conditional on win",
              "split": {"train (buckets 4-8)": count(train), "early_stop (bucket 9)": count(stop),
                        "dev (buckets 0-1)": count(dev)},
              "max_hp_changed_wins_dev": sum(bool(r["max_hp_changed"]) for r in dev),
              "results": results}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    (args.out / "summary.json").write_text(json.dumps({k: report[k] for k in ("transition_run", "excluded", "split")}
                                                      | {"dev": {n: results[n]["overall"] for n in ("baseline", "model")}}, indent=2))
    lines = [f"# combat_outcome_v1 on {args.transition_run}", "", f"Endpoint: {report['endpoint']}. {report['hp_bins']}; "
             f"wins with HP > {HP_CAP}: {overflow} (overflow bin mean {centers[-1].item():.1f}).", "",
             f"Excluded: {excluded}", "", "Split: " + json.dumps(report["split"]), "",
             "| group | n | deaths | brier b/m | logloss b/m | hp MAE b/m | bin NLL b/m | 80% cov b/m |",
             "|---|---|---|---|---|---|---|---|"]
    groups = ["ALL"] + list(results["model"]["by_group"])
    for g in groups:
        b = results["baseline"]["overall"] if g == "ALL" else results["baseline"]["by_group"][g]
        m = results["model"]["overall"] if g == "ALL" else results["model"]["by_group"][g]
        f = lambda k: f"{b.get(k, float('nan')):.3f} / {m.get(k, float('nan')):.3f}"
        lines.append(f"| {g} | {m['fights']} | {m['deaths']} | {f('brier')} | {f('log_loss')} | {f('hp_mae')} | "
                     f"{f('hp_bin_nll')} | {f('hp_80_interval_coverage')} |")
    lines += ["", "Reliability (model, dev): " + json.dumps(results["model"]["reliability"]),
              "", "Reliability (baseline, dev): " + json.dumps(results["baseline"]["reliability"]),
              "", "Model - baseline on dev (negative = model better), mean [95% CI, run-seed cluster bootstrap]:"] + [
              f"- {g}: " + ", ".join(f"{k} {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] (n {v['n']})"
                                      for k, v in boot[g].items()) for g in boot] + [
              "", f"Model: {results['model']['params']} params, best epoch {results['model']['best_epoch']} (0 = baseline init); CPU throughput "
              f"{results['model']['throughput_cpu_per_s']:.0f} states/s at batch {results['model']['throughput_batch']}."]
    lines += ["", "Notes:",
              "- 80% interval coverage uses whole 5-HP bins (true bin between the bins where the predicted CDF first "
              "reaches 0.1 and 0.9), so it is >= nominal by construction; read it per encounter, not in aggregate.",
              "- Bootstrap intervals are exploratory: they resample dev run seeds only and do not include training-seed "
              "or configuration-selection uncertainty.",
              "- Configuration history (act1-eval-mcts-v1): width 64 without baseline init, then with per-encounter "
              "linear terms, were evaluated on dev before this design (dev is not untouched); 3 configurations were "
              "then compared on the early-stop bucket 9 only.",
              "- Throughput: one timing of 20 forward passes on a shared CPU; noisy."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
