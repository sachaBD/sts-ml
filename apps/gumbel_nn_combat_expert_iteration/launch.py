"""Start a run from the repo root: python -m apps.gumbel_nn_combat_expert_iteration.launch CONFIG.toml [--scratch]
[--overwrite] (run.sh does this). runs.run owns the run directory and records the files the config reads as lineage;
the job is prepare.py."""
import argparse
from pathlib import Path
import sys

from runs import run

from .config import load
from .prepare import inputs

SCHEMA = 'gumbel_nn_combat_v1'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config', type=Path)
    p.add_argument('--scratch', action='store_true', help='a disposable run, in scratch/')
    p.add_argument('--overwrite', action='store_true', help='replace an existing run with this [run].id')
    a = p.parse_args()
    config = load(a.config.read_text())
    sys.exit(run.main([SCHEMA, config.run.id, '--live', *(f'--input={path}' for path in inputs(config)),
                       *(['--scratch'] if a.scratch else []), *(['--overwrite'] if a.overwrite else []), '--',
                       sys.executable, '-m', 'apps.gumbel_nn_combat_expert_iteration.prepare', str(a.config.resolve()),
                       '--out', '{out}']))


if __name__ == '__main__':
    main()
