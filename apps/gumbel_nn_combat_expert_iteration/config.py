"""The run's settings, typed and validated once when the config loads.

Settings have no defaults, except optional features. A count of 0 means unlimited or off (iterations,
max_train_rows, evaluation every); evaluation sims 0 plays the network alone.
[starts] draws self-play's fights from its deck pool and varies them. Each evaluation set owns its search and schedule; final_fights are
copied with the run, never played. Fights files and baselines are built offline by make_fights.py.
A run may resume with different `resumable()` settings; any other change would be a different experiment.
"""
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
import math
import sys
import tomllib
import types
from typing import get_args

from agents.combat.nn.play import Search, Workers
from agents.combat.nn.model import ModelSpec
from agents.combat.nn.export import ExportSpec
from apps.common.app import check_keys, required
from environments.combat.starts import Starts


def resumable(**kwargs):
    return field(metadata=dict(resumable=True), **kwargs)


@dataclass(frozen=True)
class Run:
    id: str
    worker: str
    seed_namespace: str
    iterations: int = resumable()
    initial_checkpoint: str | None = None
    recovery_retries: int = resumable(default=0)
    min_free_gib: float = resumable(default=0)


@dataclass(frozen=True)
class SelfPlay(Search):
    rollout_mix_decay: int = 0  # rollout_mix falls linearly to 0 at this iteration; 0 keeps it constant

    def at(self, it):
        """The search of self-play iteration `it`."""
        mix = self.rollout_mix * max(0.0, 1 - it / self.rollout_mix_decay) if self.rollout_mix_decay else self.rollout_mix
        return Search(self.sims, self.gumbel_scale, mix, self.particles)


@dataclass(frozen=True, kw_only=True)
class SelfPlayStarts(Starts):
    fights: str  # the deck pool: one fight per deck, fight_id = deck name
    per_iteration: int
    history_iterations: int = 10
    win_rate_prior_strength: float = 1.0
    sampling_concentration: float = 0.0


@dataclass(frozen=True)
class Training:
    steps: int
    batch: int
    lr: float
    replay_window: int
    max_train_rows: int = resumable()
    balanced_replay: bool
    prune_replay: bool = resumable()
    device: str
    value_batch: int = 0  # 0: shared samples; positive: one state from each of this many distinct combats
    lr_half_life: int = 0  # iterations to halve lr's distance to lr_final; 0 keeps lr constant
    lr_final: float = 0.0

    def lr_at(self, it):
        """The learning rate of training iteration `it`."""
        if not self.lr_half_life:
            return self.lr
        return self.lr_final + (self.lr - self.lr_final) * 0.5 ** (it / self.lr_half_life)


@dataclass(frozen=True)
class Model:
    kind: str
    dim: int
    layers: int
    heads: int
    gpu_tokens: int
    export_reuse: bool = resumable()
    gpu_attended: int | None = None
    gpu_mixed_precision: bool = False
    value: str = 'win'  # win: one win logit; outcome: loss/win outcome bins (agents/combat/nn/model.py)
    auxiliary: bool = False  # train HP lost, enemy HP lost and turns remaining heads
    gpu_wide_tokens: int = resumable(default=0)
    gpu_wide_attended: int | None = resumable(default=None)

    def architecture(self):
        return ModelSpec(self.kind, self.dim, self.layers, self.heads, self.value, self.auxiliary)

    def deployment(self):
        return ExportSpec(self.gpu_tokens, self.gpu_attended, self.gpu_mixed_precision)

    def wide_deployment(self):
        if self.gpu_wide_tokens:
            return ExportSpec(self.gpu_wide_tokens, self.gpu_wide_attended, self.gpu_mixed_precision)
        return None


@dataclass(frozen=True)
class EvaluationSet(Search):
    name: str
    fights: str
    at: tuple[int, ...] = resumable()
    every: int = resumable()

    def due(self, it):
        return it in self.at or bool(self.every and it % self.every == 0)


@dataclass(frozen=True)
class Evaluation:
    sets: tuple[EvaluationSet, ...]
    final_fights: str | None = None
    baselines: str | None = None
    cohorts: str | None = None


@dataclass(frozen=True)
class Config:
    run: Run
    starts: SelfPlayStarts
    search: SelfPlay
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
            item = get_args(kind)[0]
            items = required(table, f.name, name, list)
            if is_dataclass(item):
                if not all(isinstance(x, dict) for x in items):
                    sys.exit(f'[{name}].{f.name} must be a list of tables')
                values[f.name] = tuple(section(item, x, f'{name}.{f.name}[{i}]') for i, x in enumerate(items))
                continue
            if not all(isinstance(x, item) and not isinstance(x, bool) for x in items):
                sys.exit(f'[{name}].{f.name} must be a list of {item.__name__}')
            values[f.name] = tuple(items)
        else:
            values[f.name] = required(table, f.name, name, kind)
    try:
        return cls(**values)
    except ValueError as error:
        sys.exit(f'[{name}]: {error}')


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
    r, st, s, t, m, w, e = c.run, c.starts, c.search, c.training, c.model, c.workers, c.evaluation
    need(r.iterations >= 0 and r.recovery_retries >= 0 and r.min_free_gib >= 0,
         'nonnegative iterations, recovery_retries, min_free_gib')
    need(st.per_iteration > 0 and st.hp_jitter >= 0, 'starts: positive per_iteration, nonnegative hp_jitter')
    need(st.history_iterations > 0, 'starts: positive history_iterations')
    need(math.isfinite(st.win_rate_prior_strength) and st.win_rate_prior_strength > 0,
         'starts: finite positive win_rate_prior_strength')
    need(0 <= st.sampling_concentration <= 1, 'starts: sampling_concentration in [0, 1]')
    for name, search in [('search', s)] + [(f'evaluation set {x.name}', x) for x in e.sets]:
        need(search.sims >= 0 and math.isfinite(search.gumbel_scale) and search.gumbel_scale >= 0
             and 0 <= search.rollout_mix <= 1 and search.particles > 0,
             f'{name}: nonnegative sims and gumbel_scale, rollout_mix in [0, 1], positive particles')
    need(s.rollout_mix_decay >= 0, 'search: nonnegative rollout_mix_decay')
    need(t.steps > 0 and t.batch > 0 and t.lr > 0 and t.replay_window > 0 and t.max_train_rows >= 0
         and t.value_batch >= 0,
         'training: positive steps, batch, lr, replay_window; nonnegative max_train_rows, value_batch')
    need(not t.value_batch or t.max_train_rows == 0, 'value_batch requires max_train_rows = 0')
    need(t.lr_half_life >= 0 and 0 <= t.lr_final < t.lr, 'training: nonnegative lr_half_life, lr_final in [0, lr)')
    need(t.lr_half_life or t.lr_final == 0, 'training: lr_final requires lr_half_life')
    try:
        m.deployment().validate(m.architecture())
        need(m.gpu_wide_tokens >= 0, 'nonnegative gpu_wide_tokens required')
        need(m.gpu_wide_tokens or m.gpu_wide_attended is None, 'gpu_wide_attended requires gpu_wide_tokens')
        if wide := m.wide_deployment():
            wide.validate(m.architecture())
            need(m.gpu_attended is not None and m.gpu_wide_attended is not None and
                 m.gpu_wide_tokens >= m.gpu_tokens and m.gpu_wide_attended >= m.gpu_attended,
                 'wide deployment must cover narrow token and attention capacities')
    except ValueError as error:
        sys.exit(f'config: {error}')
    need(w.actors > 0 and w.slots > 0 and w.gpu_batch > 0 and w.gpu_wait_us >= 0 and w.timeout_ms > 0
         and w.shutdown_timeout_ms > 0 and w.gpu_wide_batch > 0,
         'workers: positive counts and timeouts, nonnegative gpu_wait_us')
    need(len(set(w.gpu_bank)) == len(w.gpu_bank) and all(0 < b < w.gpu_batch for b in w.gpu_bank),
         'workers.gpu_bank: distinct batches, each positive and below gpu_batch')
    need(len({s.name for s in e.sets}) == len(e.sets), 'evaluation: set names must be unique')
    for s in e.sets:
        need(bool(s.name) and s.name not in ('.', '..') and '/' not in s.name and '\0' not in s.name,
             'evaluation: set name must be a nonempty directory name')
        need(s.every >= 0 and all(i >= 0 for i in s.at),
             f'evaluation set {s.name}: nonnegative every and at')


def experiment_changes(old, new):
    """Settings that differ between two configs and may not change on resume, as [section].name."""
    def fixed(section):
        return {f.name: getattr(section, f.name) for f in fields(section) if not f.metadata.get('resumable')}

    changes = []
    for part in fields(Config):
        a, b = fixed(getattr(old, part.name)), fixed(getattr(new, part.name))
        if part.name == 'evaluation':
            for evaluation in (a, b):
                evaluation['sets'] = {s.name: fixed(s) for s in evaluation['sets']}
        changes.extend(f'[{part.name}].{key}' for key in a if a[key] != b[key])
    return sorted(changes)
