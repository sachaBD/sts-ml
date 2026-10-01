#!/usr/bin/env python3
"""CLI entrypoint; implementation lives in agents.overworld.value.learn."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.overworld.value.learn import main

if __name__ == "__main__":
    main()
