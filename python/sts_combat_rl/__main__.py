from __future__ import annotations

import argparse
import sys

from .cli.inspect import main as inspect_main
from .cli.smoke import main as smoke_main
from .data.dataset import main as dataset_main


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sts_combat_rl",
        description="Slay the Spire Combat RL Utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("smoke", help="Run neural value model smoke test")
    subparsers.add_parser("inspect", help="Inspect Parquet records as JSON")
    subparsers.add_parser("generate", help="Generate MCTS Parquet dataset")

    args, unknown = parser.parse_known_args()
    if args.command == "smoke":
        smoke_main(unknown)
    elif args.command == "inspect":
        inspect_main(unknown)
    elif args.command == "generate":
        sys.argv = [sys.argv[0]] + unknown
        dataset_main()


if __name__ == "__main__":
    main()
