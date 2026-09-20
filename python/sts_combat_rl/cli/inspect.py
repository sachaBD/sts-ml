from __future__ import annotations

import argparse
from pathlib import Path

from ..data.reader import parquet_to_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect Parquet dataset rows as JSON")
    parser.add_argument("parquet_path", type=Path, help="Path to .parquet file")
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Specific row index to inspect (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum number of rows to print"
    )
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    args = parser.parse_args(argv)

    print(
        parquet_to_json(
            args.parquet_path, row_index=args.row, limit=args.limit, indent=args.indent
        )
    )


if __name__ == "__main__":
    main()
