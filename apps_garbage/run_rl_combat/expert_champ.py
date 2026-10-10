"""Prepare (no gameplay) and inspect the recency-weighted Champ continuation.

The execution controller is deliberately gated until seed exclusions, dispatch/resume
semantics and gameplay smoke are verified. This module currently has no run subcommand.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from apps.run_rl.combat_loop import sha

DEFAULT_RUN = Path('runs/schema=combat_v4/date=2026-10-07/id=expert-champ-recency-v1')
P5 = Path('runs/schema=combat_v4/date=2026-10-06/id=rollout-expert-iteration-v1/out')


def draft_config():
    report = json.loads((P5/'strong/update3/model/train.json').read_text())
    model = (P5/'strong/update3/model').resolve()
    for name, expected in report['files_sha256'].items():
        if sha(model/name) != expected:
            raise ValueError(f'incumbent artifact mismatch: {name}')
    specs = lambda paths, generations: [dict(path=str(Path(p).resolve()), generation=g, sha256=sha(Path(p)))
                                        for p,g in zip(paths,generations)]
    anchor = specs(report['old_shards'], [0]*len(report['old_shards']))
    online = specs(report['new_shards'], [-2,-1,0])
    worker = (P5/'frozen/pv_worker').resolve()
    expected = json.loads((P5/'manifest.json').read_text())['frozen_sha256']['pv_worker']
    if sha(worker) != expected:
        raise ValueError('frozen worker mismatch')
    return dict(schema='expert_champ_recency_v1', status='SETUP_ONLY', created=datetime.now(timezone.utc).isoformat(),
                incumbent_model=str(model), incumbent_files_sha256=report['files_sha256'],
                worker=str(worker), worker_sha256=expected,
                updates=5, batch_fights=200, optimizer_steps=1000, workers=10,
                sims=2000, rollout_mix=.5, exploration=False,
                batch_size=64, anchor_states=32, online_states=32, half_life_updates=1.,
                rng_seed=0, lr=.0003, weight_decay=.01, grad_clip=1.,
                anchor_shards=anchor, initial_online_shards=online,
                monitor_fights=200, monitor_every=1, final_fights=600,
                comparison='frozen incumbent; no second training branch',
                selection='highest complete monitor wins; ties earliest update; must strictly exceed incumbent',
                final_rule='selected challenger vs frozen incumbent on fresh paired starts; no reselection',
                rollout_evaluation=True, augmentation=False, seed_manifest=None,
                launch_gates=['audit used and reserved seeds; freeze splits',
                              'implement and test fail-closed dispatch/resume controller',
                              'freeze runtime sources and verify worker flags',
                              'bounded trainer/export and gameplay smoke',
                              'announce measured runtime, log, deadline and available workers'])


def prepare(run):
    run = Path(run)
    # Never replace an existing draft or an earlier run's artifacts.
    if run.exists():
        raise ValueError(f'run already exists: {run}')
    cfg = draft_config()
    (run/'out').mkdir(parents=True)
    (run/'logs').mkdir()
    (run/'out/config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    (run/'out/state.json').write_text(json.dumps(dict(status='SETUP_ONLY', phase='Preparing controller and smoke gates',
                                                    update=0, failure=None), indent=2)+'\n')
    (run/'run.json').write_text(json.dumps(dict(status='setup_only', controller='apps.run_rl.expert_champ'), indent=2)+'\n')
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'status'])
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    a = p.parse_args()
    if a.command == 'prepare':
        prepare(a.run)
    from apps.run_rl.expert_champ_status import render
    print(render(a.run))


if __name__ == '__main__':
    main()
