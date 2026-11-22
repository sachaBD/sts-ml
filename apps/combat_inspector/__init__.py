"""Inspect replayed combat decisions using an exported checkpoint's network."""
import json
from pathlib import Path
import subprocess

from agents.combat.player.play import gpu_environment


EXECUTABLE = Path(__file__).resolve().parents[2] / 'build/main/apps/combat_inspector/combat_explain'


def request(start, *, moves=(), sims=64, particles=8, gumbel_scale=0, rollout_mix=0, considered_actions=16,
            world_draw=0):
    """One decision to search: the fight's start, the moves played since, and the search settings.

    world_draw selects which particle worlds are sampled; 0 is the draw play uses."""
    return dict(start=start, moves=list(moves),
                search=dict(sims=sims, particles=particles, gumbel_scale=gumbel_scale, rollout_mix=rollout_mix,
                            considered_actions=considered_actions, world_draw=world_draw))


def decisions(checkpoint, requests, *, cpu=False, executable=EXECUTABLE):
    """Search every request in one process, batching network calls across them; responses in request order.

    Native failures raise CalledProcessError (details in stderr)."""
    command = [str(Path(executable).resolve()), str(Path(checkpoint).resolve())] + (['--cpu'] if cpu else [])
    lines = ''.join(json.dumps(r, allow_nan=False) + '\n' for r in requests)
    completed = subprocess.run(command, input=lines, text=True, capture_output=True, check=True,
                               env=gpu_environment())
    responses = sorted((json.loads(line) for line in completed.stdout.splitlines()), key=lambda r: r['index'])
    return [{k: v for k, v in r.items() if k != 'index'} for r in responses]


def decision(checkpoint, start, *, cpu=False, executable=EXECUTABLE, **settings):
    """Return one decision and its search graph; settings as request()."""
    return decisions(checkpoint, [request(start, **settings)], cpu=cpu, executable=executable)[0]
