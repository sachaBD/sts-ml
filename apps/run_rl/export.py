#!/usr/bin/env python3
"""Export a run_rl loop directory to parquet for DuckDB (runs/README.md layout, schema run_rl_v1).

  export.py LOOP_ROOT OUT_DIR
writes OUT_DIR/results.parquet  one row per played run: part (baseline | data | eval), iter (-1 = baseline), seed,
                                boss, status, floor, fights, final_hp, seconds, cleared
       OUT_DIR/picks.parquet    one row per card reward: part, iter, seed, boss, floor, hp, max_hp, deck_size,
                                options (names, "+" = upgraded), choice, simple, source (net | simple | explore),
                                values (network after-state values, skip last; null unless the net picked)
Read with query.py views run_rl_results / run_rl_picks.
"""
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parts(root):
    if (root / "baseline" / "runs.jsonl").exists():
        yield "baseline", -1, root / "baseline" / "runs.jsonl"
    for it in sorted(root.glob("iter[0-9][0-9][0-9]")):
        i = int(re.sub(r"\D", "", it.name))
        for part in ("data", "eval"):
            f = it / part / "runs.jsonl"
            if (it / part / "summary.json").exists():  # complete parts only
                yield part, i, f


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    results, picks = [], []
    for part, i, f in parts(root):
        for line in open(f):
            r = json.loads(line)
            results.append({"part": part, "iter": i, "seed": r["seed"], "boss": r["boss"], "status": r["status"],
                            "floor": r["floor"], "fights": r["fights"], "final_hp": r["final_hp"],
                            "seconds": r["seconds"], "cleared": r["status"] == "act_complete"})
            for s in r["steps"]:
                if s["kind"] != "pick":
                    continue
                st = s["state"]
                picks.append({"part": part, "iter": i, "seed": r["seed"], "boss": r["boss"], "floor": st["floor"],
                              "hp": st["hp"], "max_hp": st["max_hp"], "deck_size": len(st["deck"]),
                              "options": [o["name"] + ("+" if o["upgraded"] else "") for o in s["options"]],
                              "choice": s["choice"], "simple": s["simple"], "source": s.get("source"),
                              "values": s.get("values")})
    pq.write_table(pa.Table.from_pylist(results), out / "results.parquet")
    pq.write_table(pa.Table.from_pylist(picks), out / "picks.parquet")
    print(f"{len(results)} runs, {len(picks)} picks -> {out}")


if __name__ == "__main__":
    main()
