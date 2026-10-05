"""Human (megacrit dump) vs agent Champ decks -> REPORT.md. Descriptive only.

  PYTHONPATH=. .venv/bin/python experiments/megacrit-champ/analyze.py \
      [--humans runs/schema=megacrit_champ_v1/date=*/id=pilot50/out/champ_starts.parquet] [--ours <champ_fights.parquet>]
"""
from __future__ import annotations

import argparse
import collections
import glob
import math
from pathlib import Path

import duckdb

from apps.megacrit_dump.champ_starts import CARDS, CURSES

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
DEF_HUMANS = str(ROOT / "runs/schema=megacrit_champ_v1/date=*/id=pilot50/out/champ_starts.parquet")
DEF_OURS = str(ROOT / "experiments/act2/champ/champ_fights.parquet")
SCALING = ["demon_form", "inflame", "spot_weakness", "limit_break"]
CURSE_NAMES = {CARDS[c].lower() for c in CURSES}
# Issues that leave base-card contents (and deck size, curses, relics) correct; only upgrade states are in doubt.
BASE_OK = {"event_card_upgrade_unknown", "bottled_relic"}


def se(k, n):
    p = k / n if n else float("nan")
    return math.sqrt(p * (1 - p) / n) if n else float("nan")


def pct(k, n):
    return "n/a" if not n else f"{100 * k / n:.0f}±{100 * se(k, n):.0f}%"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def fmt_mean(xs):
    if len(xs) < 2:
        return "n/a"
    m = mean(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return f"{m:.2f}±{sd / math.sqrt(len(xs)):.2f}"


def load(humans: str, ours: str):
    db = duckdb.connect()
    db.execute("set memory_limit='1GB'; set threads=2")
    hp = sorted(glob.glob(humans))
    if not hp:
        raise SystemExit(f"no champ_starts parquet at {humans}")
    h = db.execute("select play_id, champ_won, hp, max_hp, exact, issues, "
                   "[d.card || case when d.upgrades > 0 then '+' else '' end for d in deck] as deck, relics "
                   "from read_parquet($1)", [hp]).fetchall()
    n_raw = db.execute("select count(*) from read_parquet($1)", [ours]).fetchone()[0]
    o = db.execute("select won, hp, max_hp, deck, relics from (select distinct * exclude (fight_id, run) from read_parquet($1))",
                   [ours]).fetchall()
    humans_rows = [dict(won=r[1], hp=r[2], max_hp=r[3], exact=r[4], deck=r[6], relics=r[7],
                        reasons={i.split(":")[0] for i in r[5]}, issues=r[5]) for r in h]
    ours_rows = [dict(won=r[0], hp=r[1], max_hp=r[2], deck=r[3], relics=r[4]) for r in o]
    return humans_rows, ours_rows, n_raw


def base(c):
    return c.rstrip("+")


def deck_stats(rows):
    return dict(size=[len(r["deck"]) for r in rows], up=[sum(c.endswith("+") for c in r["deck"]) for r in rows],
                curses=[sum(base(c) in CURSE_NAMES for c in r["deck"]) for r in rows])


def overview(groups):
    out = ["| group | n | Champ win rate | mean hp/max_hp | mean hp | mean max_hp | deck size | upgraded cards | curses |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, rows, deck_ok, up_ok in groups:
        n, w = len(rows), sum(r["won"] for r in rows)
        ratio = [r["hp"] / r["max_hp"] for r in rows if r["hp"] is not None and r["max_hp"]]
        s = deck_stats(rows) if rows else None
        out.append(f"| {name} | {n} | {pct(w, n)} | {fmt_mean(ratio)} | {fmt_mean([r['hp'] for r in rows if r['hp'] is not None])} "
                   f"| {fmt_mean([r['max_hp'] for r in rows if r['max_hp'] is not None])} "
                   f"| {fmt_mean(s['size']) if deck_ok else 'n/a'} | {fmt_mean(s['up']) if up_ok else 'n/a'} "
                   f"| {fmt_mean(s['curses']) if deck_ok else 'n/a'} |")
    return "\n".join(out)


def card_sets(rows, key="deck", f=base):
    return [collections.Counter(f(c) for c in r[key]) for r in rows]


def side_stats(rows, counters, item):
    n = len(rows)
    has = [c[item] > 0 for c in counters]
    k = sum(has)
    wk = sum(r["won"] for r, h in zip(rows, has) if h)
    wo = n - k
    wno = sum(r["won"] for r, h in zip(rows, has) if not h)
    copies = sum(c[item] for c in counters) / n if n else float("nan")
    return dict(n=n, k=k, copies=copies, w_with=(wk, k), w_without=(wno, wo))


def card_table(hrows, orows, items, title_note=""):
    hc, oc = card_sets(hrows), card_sets(orows)
    out = ["| card | humans in deck | humans copies | humans win with / without | ours in deck | ours copies | ours win with / without |",
           "|---|---|---|---|---|---|---|"]
    for it in items:
        h, o = side_stats(hrows, hc, it), side_stats(orows, oc, it)
        out.append(f"| {it} | {pct(h['k'], h['n'])} | {h['copies']:.2f} | {pct(*h['w_with'])} (n={h['w_with'][1]}) / "
                   f"{pct(*h['w_without'])} (n={h['w_without'][1]}) | {pct(o['k'], o['n'])} | {o['copies']:.2f} | "
                   f"{pct(*o['w_with'])} (n={o['w_with'][1]}) / {pct(*o['w_without'])} (n={o['w_without'][1]}) |")
    return "\n".join(out)


def presence(counters, item):
    return sum(c[item] > 0 for c in counters)


def top_by_presence(counters, k):
    cnt = collections.Counter()
    for c in counters:
        cnt.update(i for i in c if c[i] > 0)
    return [i for i, _ in sorted(cnt.items(), key=lambda t: (-t[1], t[0]))[:k]]


def gap_table(hrows, orows, k=15):
    hc, oc = card_sets(hrows), card_sets(orows)
    items = set().union(*[set(c) for c in hc], *[set(c) for c in oc])
    rows = []
    for it in items:
        ph, po = presence(hc, it) / len(hrows), presence(oc, it) / len(orows)
        rows.append((abs(ph - po), it, ph, po))
    out = ["| card | humans in deck | ours in deck | humans − ours |", "|---|---|---|---|"]
    for _, it, ph, po in sorted(rows, reverse=True)[:k]:
        out.append(f"| {it} | {pct(presence(hc, it), len(hrows))} | {pct(presence(oc, it), len(orows))} | {100 * (ph - po):+.0f} pts |")
    return "\n".join(out)


def scaling(side, rows):
    cs = card_sets(rows)
    out = []
    df = [c["demon_form"] > 0 for c in cs]
    a, b = sum(df), len(rows) - sum(df)
    wa = sum(r["won"] for r, d in zip(rows, df) if d)
    wb = sum(r["won"] for r, d in zip(rows, df) if not d)
    out.append(f"| {side} | Demon Form | {a} | {pct(wa, a)} | {b} | {pct(wb, b)} |")
    buckets = collections.defaultdict(lambda: [0, 0])
    for r, c in zip(rows, cs):
        nsc = sum(c[x] for x in SCALING)
        bk = buckets[min(nsc, 3)]
        bk[0] += 1
        bk[1] += r["won"]
    lines = [f"| {side} | {'3+' if b == 3 else b} scaling cards | {buckets[b][0]} | {pct(buckets[b][1], buckets[b][0])} |"
             for b in range(4)]
    return out[0], lines


def relic_table(hrows, orows, k=20):
    hc, oc = card_sets(hrows, "relics", str), card_sets(orows, "relics", str)
    out = ["| relic | humans | ours |", "|---|---|---|"]
    for it in top_by_presence(hc, k):
        out.append(f"| {it} | {pct(presence(hc, it), len(hrows))} | {pct(presence(oc, it), len(orows))} |")
    return "\n".join(out)


def quality(allrows):
    n = len(allrows)
    ex = sum(r["exact"] for r in allrows)
    reasons_rows = collections.Counter(x for r in allrows for x in r["reasons"])
    reasons_total = collections.Counter(i.split(":")[0] for r in allrows for i in r["issues"])
    unm = collections.Counter(i.split(":", 1)[1] for r in allrows for i in r["issues"] if i.startswith("unmapped:"))
    out = [f"Exact: {ex}/{n} = {pct(ex, n)}.", "",
           "| issue reason | fights affected | issue entries |", "|---|---|---|"]
    out += [f"| {k} | {v} | {reasons_total[k]} |" for k, v in reasons_rows.most_common()]
    out += ["", f"Unmapped names: {dict(unm) or 'none'}."]
    rc = [r for r in allrows if "remove_card_not_in_deck" in r["reasons"]]
    out += ["", f"`remove_card_not_in_deck` fights that also have `deck_changing_relic_after_champ`: "
                f"{sum('deck_changing_relic_after_champ' in r['reasons'] for r in rc)}/{len(rc)} "
                "(the card was changed by a relic pickup we cannot undo)."]
    # winners and losers: why exactness is outcome-dependent
    lost = [r for r in allrows if not r["won"]]
    won = [r for r in allrows if r["won"]]
    out += ["", f"Exact among Champ losses: {pct(sum(r['exact'] for r in lost), len(lost))} (n={len(lost)}); "
                f"among wins: {pct(sum(r['exact'] for r in won), len(won))} (n={len(won)})."]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humans", default=DEF_HUMANS)
    ap.add_argument("--ours", default=DEF_OURS)
    ap.add_argument("--out", default=str(HERE / "REPORT.md"))
    a = ap.parse_args()
    allrows, ours, n_raw = load(a.humans, a.ours)
    exact = [r for r in allrows if r["exact"]]
    base_ok = [r for r in allrows if r["reasons"] <= BASE_OK]
    sc_h, sc_o = scaling("humans", base_ok), scaling("ours", ours)
    md = f"""# Human vs agent Champ decks (descriptive)

Humans: {len(allrows)} Ironclad A20 Champ fights from the megacrit pilot (50 files, 2018-10..12 and 2020-10..11).
Ours: `experiments/act2/champ/champ_fights.parquet`, {n_raw} rows, {len(ours)} after dropping rows identical in every column except `fight_id` and `run` (same seed and state replayed in several runs).
All rates are `% ± 1 binomial SE`; means are `mean ± SE`. **n is small; nothing here is a tested effect.**

Human sets: *all* ({len(allrows)}; outcome and HP are always right), *exact* ({len(exact)}; whole start state reconstructed
cleanly), *base-exact* ({len(base_ok)}; card/relic contents right, but the upgrade state of some cards is unknown because the
dump's event logs omit "+N", or a bottled relic's card is unknown). Card, relic and scaling tables use *base-exact*
(base cards, ignoring upgrades). Mean upgraded cards uses *exact* only.

**Bias to keep in mind.** A run that died at Champ is always exact. A run that won picked a boss relic afterwards, and
Astrolabe / Empty Cage / Dolly's Mirror / Whetstone / War Paint etc. cannot be undone. So exact and base-exact sets
over-represent losses and their win rate is **too low**; use *all* for the human win rate. Win rates "with/without a card"
inherit this and are only descriptive. Our agent's decks are Act-1-heavy self-play runs, not human-comparable play
(different relics/potions/pick policy), so gaps are descriptions, not skill estimates.

## 1. Overview

{overview([("humans, all", allrows, False, False), ("humans, base-exact", base_ok, True, False),
           ("humans, exact", exact, True, True), ("ours", ours, True, True)])}

## 2. Cards (base card, upgrades ignored; humans = base-exact, n={len(base_ok)}; ours n={len(ours)})

Top 40 by human presence. "copies" = mean copies over all decks. Win with/without = Champ win rate of decks with / without the card.

{card_table(base_ok, ours, top_by_presence(card_sets(base_ok), 40))}

Largest presence gaps (any card):

{gap_table(base_ok, ours)}

## 3. Strength scaling (humans = base-exact)

| side | split | n with | win with | n without | win without |
|---|---|---|---|---|---|
{sc_h[0]}
{sc_o[0]}

Scaling cards = copies of {", ".join(SCALING)}.

| side | bucket | n | win rate |
|---|---|---|---|
{chr(10).join(sc_h[1])}
{chr(10).join(sc_o[1])}

## 4. Relics (top 20 by human presence; humans = base-exact)

{relic_table(base_ok, ours)}

## 5. Reconstruction quality (all {len(allrows)} human Champ fights)

{quality(allrows)}
"""
    Path(a.out).write_text(md)
    print(md[:1500])


if __name__ == "__main__":
    main()
