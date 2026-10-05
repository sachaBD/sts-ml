"""Load the physical schema from runs/schema=megacrit_runs_v1/schema.py (the source of truth)."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "runs" / "schema=megacrit_runs_v1" / "schema.py"


def load():
    spec = spec_from_file_location("megacrit_runs_v1_schema", _PATH)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
