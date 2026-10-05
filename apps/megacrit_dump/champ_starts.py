"""Reconstruct the state at the START of each Champ fight from megacrit_runs_v1 rows -> out/champ_starts.parquet.

  PYTHONPATH=. .venv/bin/python -m runs.run megacrit_champ_v1 <id> --no-compact --input <pull run id> -- \
      .venv/bin/python apps/megacrit_dump/champ_starts.py --runs <pull run>/out/runs --out {out}

Backward from the final deck/relics: undo every change on floor >= F (the Champ floor), newest first. Anything that
cannot be undone cleanly is recorded in `issues` (exact = false), never guessed. Potions are not reconstructed.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from apps.megacrit_dump.pull import champ_fields  # noqa: E402  (tested HP indexing / F / outcome)
from apps.megacrit_dump.schema import load  # noqa: E402

SIM = Path(__file__).resolve().parents[3] / "sts_lightspeed" / "include" / "constants"
LATE = 10 ** 9  # sort key for boss relics, which are all picked after the Champ


def header_array(header: str, name: str) -> list[str]:
    text = (SIM / header).read_text()
    body = re.search(name + r"\s*\[\]\s*(?:=\s*)?\{(.*?)\}", text, re.S).group(1)
    return re.findall(r'"([^"]*)"', body)


def _ids(string_array, enum_array, header):
    s, e = header_array(header, string_array), header_array(header, enum_array)
    assert len(s) == len(e), (header, len(s), len(e))
    return dict(zip(s, e))


CARDS = _ids("cardStringIds", "cardEnumStrings", "Cards.h")
RELICS = _ids("relicIds", "relicEnumNames", "Relics.h")
POTIONS = set(header_array("Potions.h", "potionIds"))


def _curse_ids() -> set[str]:
    text = (SIM / "Cards.h").read_text()
    body = re.search(r"cardTypes\s*\[\]\s*=\s*\{(.*?)\}", text, re.S).group(1)
    types = re.findall(r"CardType::(\w+)", body)
    ids = list(CARDS)
    assert len(types) == len(ids), (len(types), len(ids))
    return {i for i, t in zip(ids, types) if t == "CURSE"}


CURSES = _curse_ids()  # curses cannot be upgraded, so an event log without "+N" is exact for them
DECK_CHANGING = {"ASTROLABE", "PANDORAS_BOX", "EMPTY_CAGE", "CALLING_BELL", "TINY_HOUSE", "DOLLYS_MIRROR", "NECRONOMICON",
                 "WAR_PAINT", "WHETSTONE"}  # the last two upgrade 2 random cards when picked up
BOTTLED = {"BOTTLED_FLAME", "BOTTLED_LIGHTNING", "BOTTLED_TORNADO"}
EVENT_KEYS = {"damage_healed", "gold_gain", "player_choice", "damage_taken", "max_hp_gain", "max_hp_loss", "event_name",
              "floor", "gold_loss", "cards_obtained", "relics_obtained", "cards_removed", "cards_upgraded",
              "cards_transformed", "relics_lost", "potions_obtained"}
CAMPFIRE_NO_DECK = {"REST", "RECALL", "DIG", "LIFT"}
SEARING = "Searing Blow"
WHY = {"remove_card", "unupgrade", "remove_relic", "remove_event_card", "add_event_card", "unupgrade_event"}


def split_card(s: str) -> tuple[str, int]:
    m = re.match(r"^(.*?)(?:\+(\d+))?$", s)
    return m.group(1), int(m.group(2) or 0)


def upgraded(name: str) -> str | None:
    base, n = split_card(name)
    if base == SEARING:
        return f"{SEARING}+{n + 1}"
    return f"{base}+1" if n == 0 else None


def kind_of(name: str) -> str | None:
    """card / relic / potion for a shop item string (None if unknown)."""
    if name in RELICS:
        return "relic"
    if name in POTIONS:
        return "potion"
    if split_card(name)[0] in CARDS:
        return "card"
    return None


class State:
    def __init__(self, deck, relics):
        self.deck = collections.Counter(deck)
        self.relics = list(relics)
        self.issues: list[str] = []

    def remove_card(self, name, why):
        if self.deck[name] > 0:
            self.deck[name] -= 1
        else:
            self.issues.append(f"remove_card_not_in_deck:{name}@{why}")

    def add_card(self, name):
        self.deck[name] += 1

    def variants(self, base):
        return [k for k, n in self.deck.items() if n > 0 and split_card(k)[0] == base]

    # Event logs name cards by id only (no "+N"), although the card may have been upgraded. These undo steps act
    # only when the deck leaves no doubt; otherwise they record an issue.
    def remove_event_card(self, name, why):
        v = self.variants(split_card(name)[0])
        if len(v) == 1:
            self.deck[v[0]] -= 1
        elif not v:
            self.issues.append(f"remove_card_not_in_deck:{name}@{why}")
        else:
            self.issues.append(f"event_card_upgrade_unknown:{name}@{why}")

    def add_event_card(self, name, why):
        if name in CURSES:
            self.deck[name] += 1
        else:  # best effort (base copy); the upgrade state of the removed card is unknowable
            self.deck[name] += 1
            self.issues.append(f"event_card_upgrade_unknown:{name}@{why}")

    def unupgrade_event(self, name, why):
        base = split_card(name)[0]
        cand = [k for k in self.variants(base) if split_card(k)[1] >= 1]
        if base != SEARING:
            cand = [k for k in cand if split_card(k)[1] == 1]
        if len(cand) == 1:
            k = cand[0]
            n = split_card(k)[1]
            self.deck[k] -= 1
            self.deck[base if n == 1 else f"{base}+{n - 1}"] += 1
        else:
            self.issues.append(f"unupgrade_no_unique_upgraded_copy:{name}@{why}")

    def unupgrade(self, name, why):
        up = upgraded(name)
        if up is not None and self.deck[up] > 0:
            self.deck[up] -= 1
            self.deck[name] += 1
        else:
            self.issues.append(f"unupgrade_no_upgraded_copy:{name}@{why}")

    def remove_relic(self, name, why):
        if name in self.relics:
            self.relics.remove(name)
            if name == "Black Blood":  # the boss swap replaced the starter relic
                self.relics.append("Burning Blood")
        else:
            self.issues.append(f"remove_relic_not_owned:{name}@{why}")
        if RELICS.get(name) in DECK_CHANGING:
            self.issues.append(f"deck_changing_relic_after_champ:{name}@{why}")

    def add_relic(self, name):
        self.relics.append(name)


def undo_ops(ev: dict, F: int, issues: list[str]):
    """(floor, priority, fn-name, args) for everything on/after the Champ floor. Priority orders one floor:
    un-upgrade first (an upgrade came after the card was obtained), then removals, then add-backs."""
    ops = []
    for c in ev.get("card_choices") or []:
        if c.get("floor", -1) >= F and c.get("picked") not in (None, "SKIP", "Singing Bowl"):
            ops.append((c["floor"], 1, "remove_card", (c["picked"],)))
    for c in ev.get("campfire_choices") or []:
        fl, key = c.get("floor", -1), c.get("key")
        if fl <= F:
            continue
        if key == "SMITH":
            ops.append((fl, 0, "unupgrade", (c.get("data"),)))
        elif key == "PURGE":
            ops.append((fl, 2, "add_card", (c.get("data"),)))
        elif key not in CAMPFIRE_NO_DECK:
            issues.append(f"unknown_campfire_key:{key}@{fl}")
    names, floors = ev.get("items_purchased") or [], ev.get("item_purchase_floors") or []
    if len(names) != len(floors):
        issues.append("purchase_floor_length_mismatch")
    for name, fl in zip(names, floors):
        if fl >= F:
            k = kind_of(name)
            if k == "card":
                ops.append((fl, 1, "remove_card", (name,)))
            elif k == "relic":
                ops.append((fl, 3, "remove_relic", (name,)))
            elif k is None:
                issues.append(f"unknown_purchase:{name}@{fl}")
    names, floors = ev.get("items_purged") or [], ev.get("items_purged_floors") or []
    if len(names) != len(floors):
        issues.append("purge_floor_length_mismatch")
    for name, fl in zip(names, floors):
        if fl >= F:
            ops.append((fl, 2, "add_card", (name,)))
    for e in ev.get("event_choices") or []:
        fl = e.get("floor", -1)
        if fl < F:
            continue
        for k in e:
            if k not in EVENT_KEYS:
                issues.append(f"unknown_event_field:{k}@{fl}")
        for n in e.get("cards_upgraded") or []:
            ops.append((fl, 0, "unupgrade_event", (n,)))
        for n in e.get("cards_obtained") or []:
            ops.append((fl, 1, "remove_event_card", (n,)))
        for n in e.get("cards_removed") or []:
            ops.append((fl, 2, "add_event_card", (n,)))
        for n in e.get("cards_transformed") or []:
            ops.append((fl, 2, "add_event_card", (n,)))
        for n in e.get("relics_obtained") or []:
            ops.append((fl, 3, "remove_relic", (n,)))
        for n in e.get("relics_lost") or []:
            ops.append((fl, 4, "add_relic", (n,)))
    for x in ev.get("relics_obtained") or []:
        if x.get("floor", -1) >= F:
            ops.append((x["floor"], 3, "remove_relic", (x["key"],)))
    for b in (ev.get("boss_relics") or [])[1:]:  # index 0 is the Act 1 boss; later ones come after the Champ
        if b.get("picked"):
            ops.append((LATE, 3, "remove_relic", (b["picked"],)))
    return ops


def dedupe_relic_ops(ops):
    """An event relic can be listed both in the event and in relics_obtained: undo each (relic, floor) once."""
    seen, out = set(), []
    for op in ops:
        if op[2] == "remove_relic":
            key = (op[0], op[3][0])
            if key in seen:
                continue
            seen.add(key)
        out.append(op)
    return out


def reconstruct(ev: dict) -> dict | None:
    c = champ_fields(ev)
    if not c["reached_champ"] or c["champ_floor"] is None:
        return None
    F = c["champ_floor"]
    issues: list[str] = []
    st = State(ev.get("master_deck") or [], ev.get("relics") or [])
    died_here = int(ev.get("floor_reached") or -1) == F and not ev.get("victory")
    ops = dedupe_relic_ops(sorted(undo_ops(ev, F, issues), key=lambda o: (-o[0], o[1])))
    if died_here:
        if ops:  # the final state already is the Champ start; undoing anything would corrupt it
            issues.append(f"died_at_champ_but_later_changes:{len(ops)}")
    else:
        for fl, _, fn, args in ops:
            getattr(st, fn)(*args, *(["boss" if fl == LATE else fl] if fn in WHY else []))
    issues += st.issues
    deck_raw = sorted(k for k, n in st.deck.items() for _ in range(n))
    relics_raw = sorted(st.relics)
    deck, unmapped = [], []
    for s in deck_raw:
        base, n = split_card(s)
        if base in CARDS:
            deck.append({"card": CARDS[base].lower(), "upgrades": n})
        else:
            unmapped.append(f"card:{s}")
    relics = []
    for r in relics_raw:
        if r in RELICS:
            relics.append(RELICS[r].lower())
        else:
            unmapped.append(f"relic:{r}")
    issues += [f"unmapped:{u}" for u in unmapped]
    issues += [f"bottled_relic:{r}" for r in relics_raw if RELICS.get(r) in BOTTLED]
    if c["hp_before_champ"] is None or c["max_hp_before_champ"] is None:
        issues.append("hp_unavailable")
    return {
        "play_id": str(ev.get("play_id")), "build_version": ev.get("build_version"),
        "timestamp": None if ev.get("timestamp") is None else int(ev["timestamp"]), "F": F,
        "champ_won": c["champ_won"], "champ_damage": c["champ_damage"], "champ_turns": c["champ_turns"],
        "hp": c["hp_before_champ"], "max_hp": c["max_hp_before_champ"],
        "deck_raw": deck_raw, "deck": sorted(deck, key=lambda d: (d["card"], d["upgrades"])),
        "relics_raw": relics_raw, "relics": sorted(relics), "unmapped": unmapped,
        "exact": not issues, "issues": issues,
    }


def run_id_of(path: Path) -> str:
    m = re.search(r"schema=([^/]+)/date=([^/]+)/id=([^/]+)", str(path.resolve()))
    return "/".join(m.groups()) if m else str(path)


def collect(dirs: list[Path]) -> list[dict]:
    rows, seen = [], set()
    for d in dirs:
        d = Path(d)
        d = d / "out" / "runs" if (d / "out" / "runs").is_dir() else d / "runs" if (d / "runs").is_dir() else d
        rid = run_id_of(d)
        for f in sorted(d.glob("part-*.parquet")):
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=500, columns=["play_id", "reached_champ", "raw"]):
                for r in batch.to_pylist():
                    if not r["reached_champ"] or (rid, r["play_id"]) in seen:
                        continue
                    seen.add((rid, r["play_id"]))
                    row = reconstruct(json.loads(r["raw"]))
                    if row:
                        rows.append({**row, "source_run_id": rid})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="megacrit_runs_v1 run dirs (or their out/ or out/runs/)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect([Path(p) for p in a.runs])
    pq.write_table(pa.Table.from_pylist(rows, schema=load("megacrit_champ_v1").CHAMP_STARTS),
                   out / "champ_starts.parquet", compression="zstd")
    reasons = collections.Counter(i.split(":")[0] for r in rows for i in r["issues"])
    summary = {"champ_fights": len(rows), "exact": sum(r["exact"] for r in rows),
               "exact_pct": round(100 * sum(r["exact"] for r in rows) / max(1, len(rows)), 1),
               "issue_reasons": dict(reasons.most_common()),
               "unmapped": dict(collections.Counter(u for r in rows for u in r["unmapped"]).most_common())}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
