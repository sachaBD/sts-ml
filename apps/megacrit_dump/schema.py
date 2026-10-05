"""Load a physical schema from runs/schema=<name>/schema.py (the source of truth)."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2] / "runs"


def load(name: str = "megacrit_runs_v1"):
    spec = spec_from_file_location(f"{name}_schema", _ROOT / f"schema={name}" / "schema.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
