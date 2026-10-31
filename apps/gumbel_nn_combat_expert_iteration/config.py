"""The run's settings, typed and validated once when the config loads. Settings have no defaults, except the
optional run features (warm start, recovery, disk reserve) and the evaluation's input sources."""
from dataclasses import MISSING, dataclass, fields
import re
import sys
import tomllib
import types
from typing import get_args

from agents.combat.nn.play import Search, Workers
from apps.common.app import check_keys, required


@dataclass(frozen=True)
class Run:
    id: str
    worker: str
    seed_namespace: str
    games_per_deck: int
    iterations: int                 # 0: until stopped
    warm_start_run: str | None = None
    recovery_retries: int = 0
    min_free_gib: float = 0


@dataclass(frozen=True)
class Training:
    steps: int
    batch: int
    lr: float
    window: int                     # train on the last `window` iterations' samples
    max_train_rows: int             # 0: all
    balanced_replay: bool
    prune_replay: bool              # keep only the window's samples and TensorRT caches
    device: str


@dataclass(frozen=True)
class Model:
    dim: int
    layers: int
    heads: int
    gpu_tokens: int                 # token count of the fixed-shape GPU models
    export_reuse: bool


@dataclass(frozen=True)
class Evaluation:
    sims: int                       # 0: the network alone
    at: tuple[int, ...]
    every: int                      # 0: only `at`
    wall_share: float               # also evaluate while validation is below this share of wall time; 0: off
    source_run: str | None = None
    directory: str | None = None
    fights_per_deck: int | None = None
    generated_per_deck: int | None = None
    starts: str | None = None
    train_starts: str | None = None
    cohorts: str | None = None
    baselines: str | None = None
    final_starts: str | None = None


@dataclass(frozen=True)
class Deck:
    name: str
    starts: str
    deck_id: str
    hp: int
    max_hp: int


@dataclass(frozen=True)
class Config:
    run: Run
    search: Search                  # self-play; evaluation plays without noise
    training: Training
    model: Model
    workers: Workers
    evaluation: Evaluation
    decks: tuple[Deck, ...]

    # Settings a resumed run may change; anything else would make it a different experiment.
    RESUMABLE = frozenset({'run.iterations', 'run.min_free_gib', 'run.recovery_retries', 'training.prune_replay',
                           'training.max_train_rows', 'model.export_reuse', 'evaluation.every', 'evaluation.wall_share'})


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
    check_keys(raw, {'run', 'search', 'training', 'model', 'workers', 'evaluation', 'deck'}, 'config')
    tables = {name: raw.get(name, {}) for name in ('run', 'search', 'training', 'model', 'workers', 'evaluation')}
    config = Config(run=section(Run, tables['run'], 'run'), search=section(Search, tables['search'], 'search'),
                    training=section(Training, tables['training'], 'training'), model=section(Model, tables['model'], 'model'),
                    workers=section(Workers, tables['workers'], 'workers'),
                    evaluation=section(Evaluation, tables['evaluation'], 'evaluation'),
                    decks=tuple(section(Deck, d, 'deck') for d in raw.get('deck', [])))
    validate(config)
    return config


def validate(c):
    def need(ok, message):
        if not ok:
            sys.exit(f'config: {message}')
    need(c.decks, 'at least one [[deck]]')
    need(len({d.name for d in c.decks}) == len(c.decks), '[[deck]] names must be unique')
    for d in c.decks:
        need(re.fullmatch(r'[a-z0-9_-]+', d.name), f'[[deck]].name {d.name!r}: use [a-z0-9_-]')
    r, s, t, m, w, e = c.run, c.search, c.training, c.model, c.workers, c.evaluation
    need(r.games_per_deck > 0 and r.iterations >= 0 and r.recovery_retries >= 0 and r.min_free_gib >= 0,
         'positive games_per_deck; nonnegative iterations, recovery_retries, min_free_gib')
    need(s.sims >= 0 and s.gumbel_scale >= 0 and 0 <= s.rollout_mix <= 1 and s.particles > 0,
         'search: nonnegative sims and gumbel_scale, rollout_mix in [0, 1], positive particles')
    need(t.steps > 0 and t.batch > 0 and t.lr > 0 and t.window > 0 and t.max_train_rows >= 0,
         'training: positive steps, batch, lr, window; nonnegative max_train_rows')
    need(m.dim > 0 and m.layers > 0 and m.heads > 0 and m.dim % m.heads == 0 and m.gpu_tokens > 0,
         'model: positive dim, layers, heads, gpu_tokens; dim divisible by heads')
    need(w.actors > 0 and w.slots > 0 and w.gpu_batch > 0 and w.gpu_wait_us >= 0 and w.timeout_ms > 0
         and w.shutdown_timeout_ms > 0, 'workers: positive counts and timeouts, nonnegative gpu_wait_us')
    need(len(set(w.gpu_bank)) == len(w.gpu_bank) and all(0 < b < w.gpu_batch for b in w.gpu_bank),
         'workers.gpu_bank: distinct batches, each positive and below gpu_batch')
    need(e.sims >= 0 and e.every >= 0 and 0 <= e.wall_share < 1 and all(i >= 0 for i in e.at),
         'evaluation: nonnegative sims, every and at; wall_share in [0, 1)')
    need(bool(e.starts) != bool(e.source_run), 'evaluation: exactly one of starts or source_run')
    need(not e.source_run or (e.directory and e.fights_per_deck), 'evaluation.source_run needs directory and fights_per_deck')


def changed(old, new):
    """Dotted names of the settings that differ between two configs."""
    def flat(c):
        return {f'{name}.{f.name}': getattr(getattr(c, name), f.name)
                for name in ('run', 'search', 'training', 'model', 'workers', 'evaluation')
                for f in fields(getattr(c, name))} | {'decks': c.decks}
    a, b = flat(old), flat(new)
    return {k for k in a if a[k] != b[k]}
