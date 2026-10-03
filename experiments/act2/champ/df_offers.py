"""How often is Demon Form offered as a card reward, and taken, before the Act 2 boss (greedy dev/fresh plays).
Usage: PYTHONPATH=. .venv/bin/python experiments/act2/champ/df_offers.py RUN_OUT_DIR..."""
import sys
from collections import Counter

from agents.overworld.value.core import iter_runs

for d in sys.argv[1:]:
    c, runs = Counter(), 0
    for _, run in iter_runs([d]):
        runs += 1
        offered = taken = False
        for s in run["steps"]:
            if s["kind"] != "pick" or s["state"].get("act", 1) > 2:
                continue
            names = [o["name"] for o in s["options"]]
            if "demon_form" in names:
                c["offers"] += 1
                offered = True
                ch = int(s["choice"])
                if ch < len(names):
                    c["taken" if names[ch] == "demon_form" else "took_other"] += 1
                    taken |= names[ch] == "demon_form"
                else:
                    c["skipped"] += 1
                c[f"act{s['state'].get('act')}_offers"] += 1
        c["runs_offered"] += offered
        c["runs_taken"] += taken
    print(d.split("id=")[1].split("/")[0], runs, "runs", dict(c))
