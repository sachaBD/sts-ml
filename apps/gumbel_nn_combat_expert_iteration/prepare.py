"""Freeze a run's inputs, then launch expert_iteration.py on them. See README.md. Called by run.sh:
  prepare.py CONFIG.toml --out DIR    freeze inputs into DIR and train
  prepare.py CONFIG.toml --inputs     print the files the run reads, one per line (its recorded lineage)

Everything the run reads is copied into OUT once: config.toml and inputs/{nn_play, starts.json, evaluation/NAME.jsonl, ...}.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tomllib

from apps.gumbel_nn_combat_expert_iteration.files import read_jsonl, snapshot, write_json
from apps.gumbel_nn_combat_expert_iteration.config import load

# Where the recipe and dashboards find each optional evaluation input.
EVALUATION_INPUTS = {'final_fights': 'FINAL_TEST_DO_NOT_USE_FOR_SELECTION.jsonl',
                     'baselines': 'baselines.json', 'cohorts': 'cohorts.json'}


def inputs(raw):
    """The files this run reads (its recorded lineage)."""
    run, starts, evaluation = (raw.get(k, {}) for k in ('run', 'starts', 'evaluation'))
    paths = [starts.get('fights'), run.get('initial_checkpoint'), *(s.get('fights') for s in evaluation.get('sets', [])),
             *(evaluation.get(k) for k in EVALUATION_INPUTS)]
    return sorted(p for p in paths if p)


def run(config_path, out):
    config = load(config_path.read_text())
    decks = read_jsonl(config.starts.fights)
    names = [f['fight_id'] for f in decks]
    if not decks or len(set(names)) != len(names) or not all(re.fullmatch(r'[a-z0-9_-]+', n) for n in names):
        sys.exit('[starts].fights: at least one, with unique fight_ids (deck names) in [a-z0-9_-]')
    if config.training.balanced_replay and len(decks) > config.training.batch:
        sys.exit('balanced_replay draws every deck in each batch: needs no more decks than [training].batch')
    frozen = out / 'inputs'
    frozen.mkdir(parents=True)  # refuse to overwrite an existing run's inputs
    snapshot(Path(config.run.worker), frozen, 'nn_play')
    (out / 'config.toml').write_bytes(config_path.read_bytes())
    write_json(frozen / 'starts.json', decks)
    (frozen / 'evaluation').mkdir()
    for evaluation in config.evaluation.sets:
        snapshot(Path(evaluation.fights), frozen / 'evaluation', f'{evaluation.name}.jsonl')
    for key, name in EVALUATION_INPUTS.items():
        if path := getattr(config.evaluation, key):
            snapshot(Path(path), frozen, name)
    if config.run.initial_checkpoint:
        snapshot(Path(config.run.initial_checkpoint), frozen, 'initial-model.pt')

    command = [sys.executable, '-m', 'apps.gumbel_nn_combat_expert_iteration.expert_iteration', str(out.resolve())]
    write_json(out / 'command.json', command)
    if config.run.recovery_retries:
        from apps.gumbel_nn_combat_expert_iteration.supervisor import launch
        launch(command, out, retries=config.run.recovery_retries)
    else:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--out', type=Path)
    action.add_argument('--inputs', action='store_true')
    args = parser.parse_args()
    if args.inputs:
        print(*inputs(tomllib.loads(args.config.read_text())), sep='\n')
    else:
        run(args.config, args.out)
