"""Inspect a replayed combat decision using an exported checkpoint's CPU network."""
import json
from pathlib import Path
import subprocess


EXECUTABLE = Path(__file__).resolve().parents[2] / 'build/main/apps/combat_inspector/combat_explain'


def decision(checkpoint, start, *, moves=(), sims=64, particles=8, gumbel_scale=0, rollout_mix=0,
             executable=EXECUTABLE):
    """Return the decision and search graph. Native failures raise CalledProcessError (details in stderr)."""
    request = dict(checkpoint=str(Path(checkpoint).resolve()), start=start, moves=list(moves),
                   search=dict(sims=sims, particles=particles, gumbel_scale=gumbel_scale, rollout_mix=rollout_mix))
    completed = subprocess.run([str(Path(executable).resolve())], input=json.dumps(request, allow_nan=False),
                               text=True, capture_output=True, check=True)
    return json.loads(completed.stdout)
