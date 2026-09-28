#!/usr/bin/env python3
"""Merge small .parquet files: see python/sts_combat_rl/compact.py.

Usage: .venv/bin/python runs/compact.py <dir> [target_mb=128]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sts_combat_rl.compact import main  # noqa: E402

main()
