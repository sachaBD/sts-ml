"""Pick rate of selected cards when offered (card rewards, Acts 1-2), per play dir.
Usage: PYTHONPATH=. .venv/bin/python experiments/act2/champ/pick_rates.py RUN_OUT_DIR..."""
import sys
from collections import Counter

from agents.overworld.value.core import iter_runs

CARDS = ["demon_form", "inflame", "spot_weakness", "limit_break", "shockwave", "metallicize", "feel_no_pain",
         "barricade", "shrug_it_off", "pommel_strike", "uppercut", "twin_strike", "perfected_strike"]
for d in sys.argv[1:]:
    off, take = Counter(), Counter()
    for _, run in iter_runs([d]):
        for s in run["steps"]:
            if s["kind"] != "pick" or s["state"].get("act", 1) > 2:
                continue
            names, ch = [o["name"] for o in s["options"]], int(s["choice"])
            for n in set(names) & set(CARDS):
                off[n] += 1
                take[n] += ch < len(names) and names[ch] == n
    print(d.split("id=")[1].split("/")[0])
    for n in CARDS:
        print(f"  {n:18s} offered {off[n]:4d}  taken {take[n] / max(off[n], 1):.2f}")
