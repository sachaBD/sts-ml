#!/usr/bin/env python3
"""Render a rundeck config template: fill {{name}} variables, then resolve {{run:SCHEMA/ID}} to the finished run id.

    .venv/bin/python experiments/elite-bench/render.py templates/TEMPLATE.toml [--var name=value ...]
        -> writes configs/TEMPLATE.toml (what run.sh launches) and prints its path

Run directories are keyed by UTC start date, which this overnight deck crosses, so templates name runs by id:
{{run:value_net_v1/elite-v3-t1}} becomes value_net_v1/<date>/elite-v3-t1 for the one finished run with that id
(it fails if there is none, more than one, or it is not `done`). Variables are filled first, so
{{run:value_net_v1/{{best}}}} works. Every {{...}} must be resolved.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def resolve(schema_id: str) -> str:
    schema, run_id = schema_id.split("/")
    matches = sorted((ROOT / "runs" / f"schema={schema}").glob(f"date=*/id={run_id}"))
    if len(matches) != 1:
        sys.exit(f"{schema_id}: need exactly one run directory, found {[str(m) for m in matches]}")
    record = json.loads((matches[0] / "run.json").read_text())
    if record["status"] != "done":
        sys.exit(f"{schema_id}: status {record['status']!r}, need done")
    return record["run_id"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("template", type=Path)
    p.add_argument("--var", action="append", default=[], help="name=value")
    a = p.parse_args()
    template = a.template if a.template.is_absolute() else HERE / a.template
    text = template.read_text()
    for item in a.var:
        name, _, value = item.partition("=")
        text = text.replace("{{" + name + "}}", value)
    text = re.sub(r"\{\{run:([a-z0-9_]+/[a-z0-9_.-]+)\}\}", lambda m: resolve(m.group(1)), text)
    if left := re.findall(r"\{\{[^}]*\}\}", text):
        sys.exit(f"unresolved in {template.name}: {left} (pass --var name=value)")
    out = HERE / "configs" / template.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
