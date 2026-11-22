"""Inspect replayed combat decisions using an exported checkpoint's network."""
import json
import os
from pathlib import Path
import subprocess

from agents.combat.player.play import gpu_environment


def _executable(name):
    build = os.environ.get('STSRL_BUILD_DIR', 'main')
    return Path(__file__).resolve().parents[2] / 'build' / build / 'apps/combat_inspector' / name


EXECUTABLE = _executable('combat_explain')


def request(start, *, moves=(), sims=64, particles=8, gumbel_scale=0, rollout_mix=0, considered_actions=16,
            world_draw=0, views=False):
    """One decision to search: the fight's start, the moves played since, and the search settings.

    world_draw selects which particle worlds are sampled; 0 is the draw play uses.
    Set views=True to attach public board views (`combat::view`) to the root and every graph node."""
    body = dict(start=start, moves=list(moves),
                search=dict(sims=sims, particles=particles, gumbel_scale=gumbel_scale, rollout_mix=rollout_mix,
                            considered_actions=considered_actions, world_draw=world_draw))
    if views:
        body['views'] = True
    return body


def decisions(checkpoint, requests, *, cpu=False, executable=None):
    """Search every request in one process, batching network calls across them; responses in request order.

    Native failures raise CalledProcessError (details in stderr)."""
    binary = _executable('combat_explain') if executable is None else Path(executable)
    command = [str(binary.resolve()), str(Path(checkpoint).resolve())] + (['--cpu'] if cpu else [])
    lines = ''.join(json.dumps(r, allow_nan=False) + '\n' for r in requests)
    completed = subprocess.run(command, input=lines, text=True, capture_output=True, check=True,
                               env=gpu_environment())
    responses = sorted((json.loads(line) for line in completed.stdout.splitlines()), key=lambda r: r['index'])
    return [{k: v for k, v in r.items() if k != 'index'} for r in responses]


def decision(checkpoint, start, *, cpu=False, executable=None, **settings):
    """Return one decision and its search graph; settings as request()."""
    return decisions(checkpoint, [request(start, **settings)], cpu=cpu, executable=executable)[0]
