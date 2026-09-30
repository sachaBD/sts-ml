"""Pre-combat outcome model: 3-seed ensemble, id<->name tables and natural presets for the GUI.

The encoding is train.batch (the training code path); tables are parsed from the sts_lightspeed headers the generator
writes ids with (apps/common/game_state.hpp).
"""
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from apps.combat_transition.train import batch, load  # noqa: E402
from apps.combat_transition.train_marginals import KINDS  # noqa: E402

RUN = ROOT / "runs/schema=combat_outcome_v1/date=2026-09-30"
CHECKPOINTS = [RUN / f"id=tpair1-co2-w32-h64-l1-d30-lr.001-s{s}/out/augmented.pt" for s in range(3)]
NATURAL = "combat_transition_v1/2026-09-29/act1-eval-mcts-a20-checked"
HEADERS = ROOT.parent / "sts_lightspeed/include/constants"
GROUPS = {"easy": ["cultist", "jaw_worm", "two_louse", "small_slimes"],
          "hard": ["gremlin_gang", "lots_of_slimes", "red_slaver", "exordium_thugs", "exordium_wildlife",
                   "blue_slaver", "looter", "large_slime", "three_louse", "two_fungi_beasts"],
          "elite": ["gremlin_nob", "lagavulin", "three_sentries"],
          "boss": ["slime_boss", "the_guardian", "hexaghost"]}


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
    def score(self, rows):
        """rows: [{"encounter", "pre"}] -> per-row dicts with per-seed and mean p_win / HP-bin distribution."""
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
    """Natural rows never trained on or used for early stopping (run_seed % 10 in 0, 1, 8; see train_marginals)."""
    rows, _ = load(NATURAL)
    return [r for r in rows if r["run_seed"] % 10 in (0, 1, 8) and r["category"] != "event"]


def calibration(ens, rows, bins=10):
    """Real vs predicted win rate on held-out natural fights: per encounter, per group, and reliability bins."""
    s = ens.score([dict(encounter=r["encounter"], pre=r["pre"]) for r in rows])
    group = {e: g for g, es in GROUPS.items() for e in es}

    def agg(idx):
        n = len(idx)
        return dict(n=n, real=sum(rows[i]["won"] for i in idx) / n if n else None,
                    pred=sum(s[i]["p_win"] for i in idx) / n if n else None,
                    seeds=[sum(s[i]["p_seeds"][k] for i in idx) / n for k in range(len(ens.nets))] if n else [])

    by_enc = {e: agg([i for i, r in enumerate(rows) if r["encounter"] == e])
              for es in GROUPS.values() for e in es if e in ens.encounters}
    by_group = {g: agg([i for i, r in enumerate(rows) if group.get(r["encounter"]) == g]) for g in GROUPS}
    rel = {}
    for g in ["all", "elite", "boss"]:
        idx = [i for i, r in enumerate(rows) if g == "all" or group.get(r["encounter"]) == g]
        rel[g] = [dict(lo=b / bins, hi=(b + 1) / bins,
                       **agg([i for i in idx if min(int(s[i]["p_win"] * bins), bins - 1) == b]))
                  for b in range(bins)]
    return dict(n=len(rows), encounters=by_enc, groups=by_group, reliability=rel)


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


def presets(rows, limit_per_encounter=6):
    """Real held-out natural pre-fight states, elites and bosses first, a few per encounter."""
    order = {"boss": 0, "elite": 1, "hard": 2, "easy": 3}
    seen, out = {}, [starter()]
    for r in sorted(rows, key=lambda r: (order.get(r["category"], 9), r["encounter"], r["run_seed"])):
        if r["category"] not in order or seen.get(r["encounter"], 0) >= limit_per_encounter:
            continue
        seen[r["encounter"]] = seen.get(r["encounter"], 0) + 1
        pre = r["pre"]
        out.append(dict(label=f"{r['encounter']} · floor {r['floor']} · seed {r['run_seed']} · "
                              f"{'won' if r['won'] else 'lost'} ({r['final_hp']} HP)",
                        encounter=r["encounter"], category=r["category"], won=r["won"], final_hp=r["final_hp"],
                        pre={k: pre[k] for k in ("deck", "relics", "potions", "hp", "max_hp", "potion_capacity")}))
    return out
