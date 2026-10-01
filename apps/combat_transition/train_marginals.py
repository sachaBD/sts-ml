#!/usr/bin/env python3
"""CLI entrypoint; implementation lives in models.combat_outcome.learn_marginals."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.combat_outcome.learn_marginals import main

if __name__ == "__main__":
    main()
