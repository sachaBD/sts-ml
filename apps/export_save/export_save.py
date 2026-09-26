"""Prototype: one stored combat_v3 fight -> IRONCLAD.autosave, resuming at the start of that fight in the game.

  PYTHONPATH=python .venv/bin/python -m apps.export_save.export_save --episode EPISODE_ID [--out PATH] [--root RUNS]

The fight is rebuilt from its source run (apps/common/replay.py) by build/<name>/export_save, which also checks
the save against the simulator's own loader. Copy the file to <Slay the Spire>/saves/IRONCLAD.autosave (back up
the one there) and press Continue. Unsupported cases (? room fights, acts 2+, ...) fail with a message.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from apps.common.replay import COLUMNS, replay_requests
from sts_combat_rl import query
from sts_combat_rl.run import REPO, RUNS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=int, required=True, help="combat_v3 episode_id (run_seed * 100 + fight_index)")
    parser.add_argument("--out", type=Path, default=Path("IRONCLAD.autosave"))
    parser.add_argument("--root", type=Path, default=RUNS, help="runs/ directory to read the fight from")
    parser.add_argument("--binary", type=Path, default=REPO / "build/main/export_save")
    args = parser.parse_args()

    # The fight's source run: the run holding its decision 0 row (combat_v3 runs hold whole act 1 runs).
    sources = query.rows(f"select * from combat_v3 where episode_id = {args.episode} and row_kind = 'decision'"
                         " and decision_index = 0", ["episode_id"], oracle=True, root=args.root)
    runs = sorted({row["run_id"] for row in sources})
    if not runs:
        sys.exit(f"episode {args.episode} not found under {args.root}")
    run_seed = args.episode // 100
    rows = query.rows(f"select * from combat_v3 where run_id = '{runs[0]}' and run_seed = {run_seed}"
                      " and row_kind = 'decision'", COLUMNS, oracle=True, root=args.root)
    request, start = replay_requests(rows, [args.episode])[args.episode]
    print(f"episode {args.episode} from {runs[0]}: fight {start['fight_index']}, floor {start['floor']}, "
          f"{start['encounter']}, hp {start['starting_hp']}/{start['starting_max_hp']}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        request_path = Path(tmp) / "request.json"
        request_path.write_text(json.dumps(request))
        done = subprocess.run([str(args.binary), str(request_path), str(args.out)])
    sys.exit(done.returncode)


if __name__ == "__main__":
    main()
