"""Load a physical schema from apps/megacrit_dump/schemas/<name>.py (the source of truth)."""
from importlib import import_module


def load(name: str = "megacrit_runs_v1"):
    return import_module(f"apps.megacrit_dump.schemas.{name}")
