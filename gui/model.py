"""Pre-combat outcome model (Acts 1-2): 3-seed ensemble, id<->name tables and natural presets for the GUI.

The encoding is train.batch (the training code path); tables are parsed from the sts_lightspeed headers the generator
writes ids with (environments/overworld/game_state.hpp).
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.combat_outcome.learn import batch  # noqa: E402
from models.combat_outcome.learn_marginals import KINDS  # noqa: E402

RUN = ROOT / "runs/schema=combat_outcome_v1/date=2026-10-02"
ARM = "natural"  # natural-only arm: better calibrated on the act-2 agent's fights than augmented (see experiments/act2/LOG.md)
CHECKPOINTS = [RUN / f"id=act2-co2-w32-h64-l1-d30-lr.001-s{s}/out/{ARM}.pt" for s in range(3)]
# Natural fights of the Act 1+2 runs of experiments/act2 (apps/combat_transition/from_overworld.py).
NATURAL = ROOT / "runs/schema=combat_transition_v1/date=2026-10-02/id=act2-overworld-fights/out"
HEADERS = ROOT.parent / "sts_lightspeed/include/constants"
CATEGORIES = ["easy", "hard", "elite", "boss"]


def group_name(act, category):
    return f"act {act} {category}"


def encounter_groups(encounters):
    """{"act N category": [encounter...]} for the model's encounters, from where they occur in the natural runs
    (most common act/category, events excluded); act 2 first, bosses first within an act."""
    rows = duckdb.sql(f"""select encounter, arg_max(act, n) act, arg_max(category, n) category from (
        select encounter, act, category, count(*) n from read_parquet('{NATURAL}/*.parquet')
        where category != 'event' group by all) group by encounter order by encounter""").fetchall()
    found = {e: (a, c) for e, a, c in rows if e in encounters}
    order = [(a, c) for a in (2, 1) for c in reversed(CATEGORIES)]
    return {group_name(a, c): sorted(e for e, ac in found.items() if ac == (a, c)) for a, c in order
            if any(ac == (a, c) for ac in found.values())}


def _array(src, name):
    body = re.search(rf"{name}\[\]\s*=?\s*{{(.*?)}};", src, re.S).group(1)
    return re.findall(r'"([^"]*)"', body) or re.findall(r"::(\w+)", body)


def tables():
    """Cards (with color/type/rarity), relics and potions, indexed by the simulator's enum ids."""
    c = (HEADERS / "Cards.h").read_text()
    enums, names = _array(c, "cardEnumStrings"), _array(c, "cardNames")
    colors, types, rar = (re.findall(r"::(\w+)", re.search(rf"{n}\[\]\s*=\s*{{(.*?)}};", c, re.S).group(1))
                          for n in ("cardColors", "cardTypes", "cardRarities"))
    cards = [dict(id=i, key=enums[i].lower(), name=names[i], color=colors[i].lower(), type=types[i].lower(),
                  rarity=rar[i].lower()) for i in range(1, len(enums))]
    r = (HEADERS / "Relics.h").read_text()
    relics = [dict(id=i, key=k.lower(), name=n) for i, (k, n) in
              enumerate(zip(_array(r, "relicEnumNames"), _array(r, "relicNames"))) if k != "INVALID"]
    p = (HEADERS / "Potions.h").read_text()
    potions = [dict(id=i, key=k.lower(), name=n) for i, (k, n) in
               enumerate(zip(_array(p, "potionEnumNames"), _array(p, "potionNames"))) if i >= 2]
    return dict(cards=cards, relics=relics, potions=potions)


class Ensemble:
    def __init__(self, paths=CHECKPOINTS):
        self.nets, self.encounters, self.centers = [], None, None
        for path in paths:
            c = torch.load(path, map_location="cpu")
            net = KINDS[c["kind"]](**c["args"])
            net.load_state_dict(c["state_dict"])
            self.nets.append(net.eval())
            self.encounters = self.encounters or c["encounters"]
            self.centers = c["centers"]
            assert c["encounters"] == self.encounters and c["centers"] == self.centers

    @torch.no_grad()
    def score(self, rows, chunk=4096):
        """rows: [{"encounter", "pre"}] -> per-row dicts with per-seed and mean p_win / HP-bin distribution."""
        out = []
        for start in range(0, len(rows), chunk):
            out += self._score(rows[start:start + chunk])
        return out

    def _score(self, rows):
        b, _, _ = batch([dict(r, won=0, final_hp=0) for r in rows], self.encounters, "cpu")
        centers = torch.tensor(self.centers)
        per = []
        for net in self.nets:
            w, h = net(b)
            per.append((torch.sigmoid(w), torch.softmax(h, -1)))
        p = torch.stack([x[0] for x in per], 1)       # B x seeds
        h = torch.stack([x[1] for x in per], 1)       # B x seeds x bins
        hp_mean = (h * centers).sum(-1)                # B x seeds
        ev = p * hp_mean
        out = []
        for i in range(len(rows)):
            out.append(dict(p_win=p[i].mean().item(), p_seeds=p[i].tolist(),
                            hp_if_win=hp_mean[i].mean().item(), expected_hp=ev[i].mean().item(),
                            ev_seeds=ev[i].tolist(), hp_bins=h[i].mean(0).tolist()))
        return out


def natural_dev():
    """Natural rows never trained on or used for early stopping (run_seed % 10 in 0, 1, 8; see train_marginals),
    with a verified pre-fight state (replay = 'ok'), events excluded."""
    t = pq.read_table(sorted(NATURAL.glob("*.parquet")), filters=[("bucket", "in", [0, 1, 8]), ("replay", "=", "ok"), ("category", "!=", "event")],
                      columns=["source_run_id", "run_seed", "act", "floor", "encounter", "category", "won", "final_hp",
                               "pre"])
    return t.to_pylist()


def calibration(ens, rows, groups, bins=10):
    """Real vs predicted win rate on held-out natural fights: per encounter, per group, and reliability bins."""
    s = ens.score([dict(encounter=r["encounter"], pre=r["pre"]) for r in rows])
    group = {e: g for g, es in groups.items() for e in es}

    def agg(idx):
        n = len(idx)
        return dict(n=n, real=sum(rows[i]["won"] for i in idx) / n if n else None,
                    pred=sum(s[i]["p_win"] for i in idx) / n if n else None,
                    seeds=[sum(s[i]["p_seeds"][k] for i in idx) / n for k in range(len(ens.nets))] if n else [])

    by_enc = defaultdict(list)
    for i, r in enumerate(rows):
        by_enc[r["encounter"]].append(i)
    by_group = defaultdict(list)
    for i, r in enumerate(rows):
        if r["encounter"] in group:
            by_group[group[r["encounter"]]].append(i)
    rel = {}
    for g in ["all", group_name(1, "boss"), group_name(2, "boss")]:
        idx = [i for i, r in enumerate(rows) if g == "all" or group.get(r["encounter"]) == g]
        rel[g] = [dict(lo=b / bins, hi=(b + 1) / bins,
                       **agg([i for i in idx if min(int(s[i]["p_win"] * bins), bins - 1) == b]))
                  for b in range(bins)]
    return dict(n=len(rows), encounters={e: agg(by_enc[e]) for es in groups.values() for e in es},
                groups={g: agg(by_group[g]) for g in groups}, reliability=rel)


def starter():
    """Ironclad A20 starting state: 5 Strike, 4 Defend, Bash, Ascender's Bane, Burning Blood, 68/75 HP, 2 potion slots.
    Ids come from the header tables; 68/75 matches the floor-1 natural rows."""
    t = tables()
    card = {c["key"]: c["id"] for c in t["cards"]}
    relic = {r["key"]: r["id"] for r in t["relics"]}
    deck = [dict(card_id=card[k], upgraded=0, misc=0, name=k)
            for k in ["ascenders_bane"] + ["strike_red"] * 5 + ["defend_red"] * 4 + ["bash"]]
    return dict(label="Starter deck (A20, 68/75 HP)", encounter="jaw_worm", category="starter", won=None,
                final_hp=None, pre=dict(deck=deck, relics=[dict(relic_id=relic["burning_blood"], data=0,
                                                                name="burning_blood")],
                                        potions=[], hp=68, max_hp=75, potion_capacity=2))


def presets(rows, groups, limit_per_encounter=8):
    """Real held-out natural pre-fight states, a few per encounter, in the groups' order (Act 2 bosses first).
    One per run seed and encounter, spread over the dev runs."""
    order = {e: i for i, e in enumerate(e for es in groups.values() for e in es)}
    picked = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["run_seed"] * 7919 % 10007, r["source_run_id"])):
        e = r["encounter"]
        if e in order and len(picked[e]) < limit_per_encounter and all(x["run_seed"] != r["run_seed"] for x in picked[e]):
            picked[e].append(r)
    out = [starter()]
    for e in sorted(picked, key=order.get):
        for r in sorted(picked[e], key=lambda r: r["floor"]):
            pre = r["pre"]
            out.append(dict(label=f"{r['encounter']} · floor {r['floor']} · seed {r['run_seed']} · "
                                  f"{'won' if r['won'] else 'lost'} ({r['final_hp']} HP)",
                            encounter=e, category=f"act {r['act']} {r['category']}", won=r["won"],
                            final_hp=r["final_hp"],
                            pre={k: pre[k] for k in ("deck", "relics", "potions", "hp", "max_hp", "potion_capacity")}))
    return out
