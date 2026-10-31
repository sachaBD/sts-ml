"""Freeze a run's inputs, then launch expert_iteration.py on them. See README.md.

Everything the run reads is copied into OUT once: config.toml and inputs/{nn_play, starts.json, test.jsonl, ...}.
"""
import json
from pathlib import Path
import re
import subprocess
import sys

from apps.common.app import main, snapshot, write_json
from apps.gumbel_nn_combat_expert_iteration.config import load

# Where the recipe and dashboards find each optional evaluation input.
EVALUATION_INPUTS = {'in_sample_fights': 'test-in.jsonl', 'final_fights': 'FINAL_TEST_DO_NOT_USE_FOR_SELECTION.jsonl',
                     'baselines': 'baselines.json', 'cohorts': 'cohorts.json'}


def inputs(raw):
    """The files this run reads (lineage for apps/common/launch.sh)."""
    run, training, evaluation = (raw.get(k, {}) for k in ('run', 'training', 'evaluation'))
    paths = [training.get('fights'), run.get('initial_checkpoint'), evaluation.get('fights'),
             *(evaluation.get(k) for k in EVALUATION_INPUTS)]
    return sorted(p for p in paths if p)


def read_fights(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run(_raw, config_path, out):
    config = load(config_path.read_text())
    decks = read_fights(config.training.fights)
    names = [f['fight_id'] for f in decks]
    if not decks or len(set(names)) != len(names) or not all(re.fullmatch(r'[a-z0-9_-]+', n) for n in names):
        sys.exit('training fights: at least one, with unique fight_ids (deck names) in [a-z0-9_-]')
    frozen = out / 'inputs'
    frozen.mkdir(parents=True)  # refuse to overwrite an existing run's inputs
    snapshot(Path(config.run.worker), frozen, 'nn_play')
    (out / 'config.toml').write_bytes(config_path.read_bytes())
    write_json(frozen / 'starts.json', decks)
    snapshot(Path(config.evaluation.fights), frozen, 'test.jsonl')
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
    main(run, inputs)
