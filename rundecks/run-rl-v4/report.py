"""Stage 3: curve + paired fresh-seed comparison -> rundecks/run-rl-v4/RESULTS.md; parquet export + run.json."""
import json, subprocess, sys
from pathlib import Path
R = Path("runs/schema=run_rl_v1/date=2026-10-01/id=v4-all")
E = Path("runs/schema=run_rl_v1/date=2026-10-01/id=event-h0-test/out")
L = lambda p: {r["seed"]: r for r in map(json.loads, open(p))}
won = lambda r: r["status"] == "act_complete"
def pair(a, b):
    s = sorted(set(a) & set(b)); d = [won(a[k]) - won(b[k]) for k in s]; n = len(d); m = sum(d) / n
    se = (sum((x - m) ** 2 for x in d) / (n - 1) / n) ** .5
    return f"{sum(won(a[k]) for k in s) / n:.3f} vs {sum(won(b[k]) for k in s) / n:.3f}: {m:+.3f} ± {se:.3f} (n={n})"
v4 = L(R / "out/fresh-iter001/runs.jsonl")
lines = ["# run-rl-v4 results (written by report.py)", "", "## Eval curve (500 eval seeds; SimpleAgent baseline in the same run)",
         "", "```", (R / "out/curve.tsv").read_text().strip(), "```", "", "## Fresh seeds 830000000000+ (paired)", "",
         f"- v4 iter1 (all decisions) vs v3 (SimpleAgent events + Neow): {pair(v4, L(E / 'no-event/runs.jsonl'))}",
         f"- v4 iter1 vs v3 with events + Neow by lookahead: {pair(v4, L(E / 'event-neow/runs.jsonl'))}"]
Path("rundecks/run-rl-v4/RESULTS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
subprocess.run([sys.executable, "apps/run_rl/export.py", str(R / "out"), str(R / "out")], check=True)
curve = [l.split("\t") for l in (R / "out/curve.tsv").read_text().splitlines()[1:]]
json.dump({"run_id": "run_rl_v1/2026-10-01/v4-all", "schema": "run_rl_v1", "scratch": False, "status": "done",
           "note": "Real-run RL loop v4: every decision by V (cards, rest, path, shop; events + Neow by sampled lookahead h=0, 8 samples). Init v3 iter003; 2 iterations. rundecks/run-rl-v4 (hand-written run.json).",
           "inputs": ["run_rl_v1/2026-10-01/v3-shop", "run_rl_v1/2026-10-01/v2-rest-path"],
           "command": ["bash", "rundecks/run-rl-v4/loop.sh"],
           "summary": {"curve": [{"iter": int(c[0]), "net_clear": float(c[1]), "diff": float(c[3]), "se": float(c[4])} for c in curve]}},
          open(R / "run.json", "w"), indent=2)
