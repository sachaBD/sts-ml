"""The run's settings, typed and validated once when the config loads.

Settings have no defaults, except optional features. A count of 0 means unlimited or off (iterations,
max_train_rows, evaluation every and wall_share); evaluation sims 0 plays the network alone.
Training fights are one per deck, fight_id = deck name. Evaluation plays [search] without noise; final_fights
are copied with the run, never played. Fights files and baselines are built offline by make_fights.py.
A run may resume with different `resumable()` settings; any other change would be a different experiment.
"""
from dataclasses import MISSING, dataclass, field, fields
import sys
import tomllib
import types
from typing import get_args

from agents.combat.nn.play import Search, Workers
from apps.common.app import check_keys, required


def resumable(**kwargs):
    return field(metadata=dict(resumable=True), **kwargs)


@dataclass(frozen=True)
class Run:
    id: str
    worker: str
    seed_namespace: str
    games_per_deck: int
    iterations: int = resumable()
    initial_checkpoint: str | None = None
    recovery_retries: int = resumable(default=0)
    min_free_gib: float = resumable(default=0)


@dataclass(frozen=True)
class Training:
    fights: str
    steps: int
    batch: int
    lr: float
    replay_window: int
    max_train_rows: int = resumable()
    balanced_replay: bool
    prune_replay: bool = resumable()
    device: str


@dataclass(frozen=True)
class Model:
    dim: int
    layers: int
    heads: int
    gpu_tokens: int
    export_reuse: bool = resumable()
    kind: str = 'net'
    gpu_attended: int | None = None


@dataclass(frozen=True)
class Evaluation:
    sims: int
    at: tuple[int, ...]
    every: int = resumable()
    wall_share: float = resumable()
    fights: str
    in_sample_fights: str | None = None
    final_fights: str | None = None
    baselines: str | None = None
    cohorts: str | None = None


@dataclass(frozen=True)
class Config:
    run: Run
    search: Search
    training: Training
    model: Model
    workers: Workers
    evaluation: Evaluation


def section(cls, table, name):
    check_keys(table, {f.name for f in fields(cls)}, name)
    values = {}
    for f in fields(cls):
        if f.name not in table:
            if f.default is MISSING:
                sys.exit(f'missing [{name}].{f.name}')
            continue
        kind = f.type
        if isinstance(kind, types.UnionType):  # optional: X | None
            kind = next(t for t in get_args(kind) if t is not type(None))
        if getattr(kind, '__origin__', None) is tuple:
            items = required(table, f.name, name, list)
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in items):
                sys.exit(f'[{name}].{f.name} must be a list of integers')
            values[f.name] = tuple(items)
        else:
            values[f.name] = required(table, f.name, name, kind)
    return cls(**values)


def load(text):
    raw = tomllib.loads(text)
    check_keys(raw, {f.name for f in fields(Config)}, 'config')
    config = Config(**{f.name: section(f.type, raw.get(f.name, {}), f.name) for f in fields(Config)})
    validate(config)
    return config


def validate(c):
    def need(ok, message):
        if not ok:
            sys.exit(f'config: {message}')
    r, s, t, m, w, e = c.run, c.search, c.training, c.model, c.workers, c.evaluation
    need(r.games_per_deck > 0 and r.iterations >= 0 and r.recovery_retries >= 0 and r.min_free_gib >= 0,
         'positive games_per_deck; nonnegative iterations, recovery_retries, min_free_gib')
    need(s.sims >= 0 and s.gumbel_scale >= 0 and 0 <= s.rollout_mix <= 1 and s.particles > 0,
         'search: nonnegative sims and gumbel_scale, rollout_mix in [0, 1], positive particles')
    need(t.steps > 0 and t.batch > 0 and t.lr > 0 and t.replay_window > 0 and t.max_train_rows >= 0,
         'training: positive steps, batch, lr, replay_window; nonnegative max_train_rows')
    need(m.dim > 0 and m.layers > 0 and m.heads > 0 and m.dim % m.heads == 0 and m.gpu_tokens > 0,
         'model: positive dim, layers, heads, gpu_tokens; dim divisible by heads')
    need(m.kind in ('net', 'entity'), 'model.kind: net or entity')
    need(m.kind != 'entity' or (m.gpu_attended is not None and 0 < m.gpu_attended <= m.gpu_tokens and not m.export_reuse),
         'entity model: 0 < gpu_attended <= gpu_tokens; export_reuse must be false')
    need(m.kind != 'net' or m.gpu_attended is None, 'gpu_attended is only supported for entity models')
    need(w.actors > 0 and w.slots > 0 and w.gpu_batch > 0 and w.gpu_wait_us >= 0 and w.timeout_ms > 0
         and w.shutdown_timeout_ms > 0, 'workers: positive counts and timeouts, nonnegative gpu_wait_us')
    need(len(set(w.gpu_bank)) == len(w.gpu_bank) and all(0 < b < w.gpu_batch for b in w.gpu_bank),
         'workers.gpu_bank: distinct batches, each positive and below gpu_batch')
    need(e.sims >= 0 and e.every >= 0 and 0 <= e.wall_share < 1 and all(i >= 0 for i in e.at),
         'evaluation: nonnegative sims, every and at; wall_share in [0, 1)')


def experiment_changes(old, new):
    """Settings that differ between two configs and may not change on resume, as [section].name."""
    return sorted(f'[{part.name}].{f.name}' for part in fields(Config) for f in fields(part.type)
                  if not f.metadata.get('resumable') and getattr(getattr(old, part.name), f.name) != getattr(getattr(new, part.name), f.name))
