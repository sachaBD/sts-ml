"""Multi-deck corpus controller: teacher entry, zero-shot, quota'd collection, mixed-deck training, monitor evals.

`run` is resumable (rerun the same command; combat_loop stage markers skip finished stages). `out/control.json`
is re-read every update (`stop: true` exits cleanly). `status` is a read-only dashboard; `freeze` snapshots the
latest model and the graduated list for S3. See experiments/human-deck-corpus/SPEC_CONTROLLER.md.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import re
import shutil
import subprocess
import statistics
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from apps.run_rl.combat_loop import PY, encode, play, read_results, sha, stage, write_starts
from apps.run_rl.single_deck import generate, score

DEFAULTS = dict(fights_per_update=400, min_fights_per_deck=10, wave_every=6, wave_size=8, eval_every=4,
                replay_fights_per_deck=300, teacher_per_deck=50, epochs=3, lr=3e-4, stop=False)
BOOTSTRAP = dict(epochs=10, lr=1e-3)
TAPER, PARK_FIGHTS, PARK_P, MAX_INCOMPLETE = 5, 150, 0.05, 0.05
STATES_PER_FIGHT = 64
EXCLUDE_DEFAULT = [Path('runs/schema=human_champ_bench_v1/date=2026-10-05/id=v1/out/starts.parquet'),
                   Path('runs/schema=combat_v4/date=2026-10-05/id=human-deck-library-v1/out/starts.parquet')]


# ---- pure helpers (unit tested) -------------------------------------------------------------------------------
def jwrite(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def smoothed_p(batches, last=2):
    """Beta(1,1)-smoothed win rate over the last `last` collection batches (zero-shot batch included)."""
    recent = batches[-last:]
    return (sum(b['wins'] for b in recent) + 1) / (sum(b['completed'] for b in recent) + 2)


def fight_seconds(deck):
    """Mean seconds per recent learner fight; screen MCTS seconds / 5 before any learner fight."""
    recent = [b for b in deck['batches'] if b['kind'] == 'learner'][-2:]
    done = sum(b['completed'] for b in recent)
    if done:
        return max(sum(b['seconds'] for b in recent) / done, 1e-3)
    return max(deck['screen_seconds'] / 5, 1e-3)


def allocate(budget, floor, decks):
    """decks: [(id, graduated, weight)]. Everyone gets `floor`; the rest goes to non-graduated decks ∝ weight
    by largest remainder (ties by id). If floors exceed the budget the floors still apply."""
    quotas = {d: floor for d, _, _ in decks}
    active = [(d, w) for d, g, w in decks if not g]
    extra, total = budget - floor * len(decks), sum(w for _, w in active)
    if extra <= 0 or total <= 0:
        return quotas
    exact = {d: extra * w / total for d, w in active}
    for d in exact:
        quotas[d] += int(exact[d])
    left = extra - sum(int(v) for v in exact.values())
    for d in sorted(exact, key=lambda d: (-(exact[d] - int(exact[d])), d))[:left]:
        quotas[d] += 1
    return quotas


def teacher_count(age, tpd):
    return max(0, tpd - TAPER * age)


def training_ids(deck, k, ctl):
    """fight_id -> shard for one deck: last R completed learner fights + age-tapered deterministic teacher subset."""
    out = {}
    learner = [(fid, b['shard']) for b in deck['batches'] if b['kind'] == 'learner' for fid in b['ids']]
    for fid, shard in learner[-ctl['replay_fights_per_deck']:]:
        out[fid] = shard
    if ctl.get('taper_after_mastery'):
        # Taper the MCTS teacher only once the learner matches it on this deck's monitor seeds.
        start = deck.get('mastered_at')
        n = ctl['teacher_per_deck'] if start is None else teacher_count(k - start, ctl['teacher_per_deck'])
    else:
        n = teacher_count(k - deck['entry_update'], ctl['teacher_per_deck'])
    pool = sorted(deck['teacher_pool'], key=lambda f: hashlib.sha256(f.encode()).hexdigest())
    for fid in pool[:n]:
        out[fid] = deck['teacher_pool'][fid]
    return out


def next_status(status, ev, p, learner_fights, threshold):
    """Lifecycle after evals -> (new status, reason or None). `ev` is this update's monitor result or None."""
    if status in ('pending', 'software-failed'):
        return status, None
    if ev and ev['n']:
        good = ev['wins'] >= ev['mcts_wins'] and ev['wins'] >= threshold
        if good and status != 'graduated':
            return 'graduated', f'monitor {ev["wins"]}/{ev["n"]} >= MCTS {ev["mcts_wins"]} and >= {threshold}'
        if not good and status == 'graduated':
            return 'active', f'monitor {ev["wins"]}/{ev["n"]} below MCTS {ev["mcts_wins"]} or {threshold}'
    if status == 'active' and learner_fights >= PARK_FIGHTS and p < PARK_P:
        return 'parked', f'p={p:.3f} < {PARK_P} after {learner_fights} learner fights'
    return status, None


def macro(decks):
    """Latest eval per deck -> macro means and paired gap with SE clustered by deck."""
    rows = [d['evals'][-1] for d in decks if d['evals'] and d['evals'][-1]['n']]
    if not rows:
        return None
    gaps = [r['wins'] / r['n'] - r['mcts_wins'] / r['n'] for r in rows]
    mean = sum(gaps) / len(gaps)
    se = statistics.stdev(gaps) / math.sqrt(len(gaps)) if len(gaps) > 1 else None
    return dict(decks=len(rows), learned=sum(r['wins'] / r['n'] for r in rows) / len(rows),
                mcts=sum(r['mcts_wins'] / r['n'] for r in rows) / len(rows), gap=mean, gap_se=se)


def subset(sources, ids, dest):
    """Copy only `ids` of fights-0/search-0 from several play dirs into one pseudo play dir (for pv.data encode)."""
    dest.mkdir(parents=True, exist_ok=True)
    keep = pa.array(sorted(ids))
    for name in ('fights-0.parquet', 'search-0.parquet'):
        tables = []
        for s in sources:  # ParquetFile: no hive partition columns from runs/schema=.../ paths
            t = pq.ParquetFile(s / name).read()
            tables.append(t.filter(pc.is_in(t['fight_id'], keep)))
        tables = [t.cast(tables[0].schema) for t in tables]
        pq.write_table(pa.concat_tables(tables), dest / name, compression='zstd')


# ---- controller -----------------------------------------------------------------------------------------------
class Controller:
    def __init__(self, a):
        self.a = a
        self.out = a.out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.lock = (self.out / 'controller.lock').open('a')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        (self.out / 'logs').mkdir(exist_ok=True)
        self.ns = a.namespace
        deck_list = json.loads(a.decks.read_text())
        self.decks = {d['deck_id']: d for d in deck_list}
        self.order = [d['deck_id'] for d in deck_list]
        if len(self.decks) != len(deck_list):
            raise ValueError('duplicate deck_id in decks.json')
        self.threshold = math.ceil(0.9 * a.monitor)
        config = dict(worker_sha=sha(a.worker), namespace=a.namespace, device=a.device,
                      sims=a.sims, teacher_sims=a.teacher_sims, monitor=a.monitor, final=a.final,
                      init_sha=sha(a.init / 'model.pt') if a.init else None,
                      exclude={str(p): sha(p) for p in a.exclude_starts}, bootstrap=BOOTSTRAP)
        cp = self.out / 'config.json'
        if cp.exists() and json.loads(cp.read_text()) != config:
            raise ValueError('config changed on resume (only control.json, workers and --updates may change)')
        jwrite(cp, config)
        self.worker = self.out / 'frozen/pv_worker'
        if not self.worker.exists():
            self.worker.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(a.worker, self.worker)
        if sha(self.worker) != config['worker_sha']:
            raise ValueError('frozen worker checksum mismatch')
        self.excluded = set()
        for path in a.exclude_starts:
            self.excluded |= {r['start']['seed'] for r in pq.read_table(path, columns=['start']).to_pylist()}
        for d in self.decks.values():
            self.excluded.add(d['start']['start']['seed'])
        sp = self.out / 'state.json'
        if sp.exists():
            for d, rec in json.loads(sp.read_text())['decks'].items():
                if d not in self.decks:
                    raise ValueError(f'entered deck {d} missing from decks file on resume')
        hist = self.out / 'decks-history.json'
        seen = json.loads(hist.read_text()) if hist.exists() else []
        if not seen or seen[-1]['sha'] != sha(a.decks):
            seen.append(dict(sha=sha(a.decks), path=str(a.decks.resolve()), at=time.time()))
            jwrite(hist, seen)
        self.state = json.loads(sp.read_text()) if sp.exists() else dict(
            k=-1, started=time.time(), phase='init', stage=None, decks={}, model=None, model_k=None, plan=None,
            allocations={}, updates={}, log=[], control=None)
        self.control_seen = self.state.get('control')

    # -- plumbing
    def log(self, msg):
        line = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {msg}'
        print(line, flush=True)
        with (self.out / 'logs/controller.log').open('a') as f:
            f.write(line + '\n')

    def save(self, phase=None, stage_name=None):
        if phase:
            self.state['phase'] = phase
        self.state['stage'] = stage_name
        self.state['updated'] = time.time()
        jwrite(self.out / 'state.json', self.state)

    def load_control(self):
        path = self.out / 'control.json'
        if not path.exists():
            jwrite(path, DEFAULTS)
        ctl = {**DEFAULTS, **json.loads(path.read_text())}
        if ctl != self.control_seen:
            changed = {k: v for k, v in ctl.items() if not self.control_seen or self.control_seen.get(k) != v}
            self.log(f'control.json now {ctl}' if not self.control_seen else f'control.json changed: {changed}')
            self.control_seen = self.state['control'] = ctl
        return ctl

    def rows(self, deck, split, n):
        """Deterministic, deck-independent starts: seed from sha256(ns:deck:split:i), skipping the static excluded set."""
        return generate(self.decks[deck]['start'], f'{self.ns}:{deck}', split, n, set(self.excluded)) if n > 0 else []

    def monitor_ids(self, deck):
        return [f'{self.ns}:{deck}:monitor:{i}' for i in range(self.a.monitor)]

    def entered(self, failed=False):
        return [d for d in self.order if d in self.state['decks']
                and (failed or self.state['decks'][d]['status'] != 'software-failed')]

    def split_manifest(self):
        mapping = {}
        for d in self.entered(failed=True):
            rec = self.state['decks'][d]
            if rec.get('encode_failed'):  # nothing of this deck is ever trained or validated
                continue
            mapping.update({f: 'train' for f in rec['teacher_pool']})
            mapping.update({f: 'val' for f in self.monitor_ids(d)})
            for b in rec['batches']:
                if b['kind'] == 'learner':
                    mapping.update({f'{self.ns}:{d}:learner-{b["k"]}:{i}': 'train' for i in range(b['attempted'])})
        jwrite(self.out / 'split.json', mapping)
        return self.out / 'split.json'

    def encode_failure(self, name):
        """Reason for a failed per-deck encode stage: its 'unsupported' log lines, else the last log line."""
        log = self.out / 'logs' / f'{name}.log'
        lines = log.read_text().splitlines() if log.exists() else []
        hit = [l.strip() for l in lines if 'unsupported' in l.lower()]
        return 'encoder ' + ('; '.join(dict.fromkeys(hit))[:300] if hit else (lines[-1].strip()[:300] if lines else 'failed'))

    # -- stages
    def entry(self, k, ids, ctl):
        """Teacher top-up + monitor reference for entering decks (one play stage), shard encodes, zero-shot eval."""
        rows, tops = [], {}
        for d in ids:
            tops[d] = max(0, ctl['teacher_per_deck'] - len(self.decks[d]['teacher_fight_ids']))
            rows += self.rows(d, 'teacher', tops[d]) + self.rows(d, 'monitor', self.a.monitor)
        self.save(stage_name=f'enter-{k}')
        write_starts(rows, self.out / f'enter-{k}-starts.parquet')
        ref = play(self.out, f'enter-{k}', self.out / f'enter-{k}-starts.parquet', None, self.worker,
                   self.a.teacher_sims, self.a.workers, teacher=True)
        results = read_results(ref)
        done = lambda fid, res: res.get(fid, {}).get('status') == 'completed'
        new = {}
        for d in ids:
            spec = self.decks[d]
            screen_res = {}
            for run in spec['teacher_runs']:
                screen_res.update(read_results(run))
            screen = [f for f in spec['teacher_fight_ids'] if done(f, screen_res)]
            top = [f'{self.ns}:{d}:teacher:{i}' for i in range(tops[d]) if done(f'{self.ns}:{d}:teacher:{i}', results)]
            mon = [f for f in self.monitor_ids(d) if done(f, results)]
            pool, shard, failure = {}, None, None
            name = None
            try:
                if screen:
                    name = f'shard-screen-{d}'
                    if not (self.out / (name + '.done')).exists():
                        subset([Path(r) for r in spec['teacher_runs']], screen, self.out / f'subset-{name}')
                    path = encode(self.out, name, self.out / f'subset-{name}', self.worker)
                    pool.update({f: str(path) for f in screen})
                shard = None
                if top or mon:
                    name = f'shard-teacher-{d}'
                    if not (self.out / (name + '.done')).exists():
                        subset([ref], top + mon, self.out / f'subset-{name}')
                    shard = str(encode(self.out, name, self.out / f'subset-{name}', self.worker))
                    pool.update({f: shard for f in top})
            except subprocess.CalledProcessError:
                failure = self.encode_failure(name)
                pool, shard = {}, None
                self.log(f'{d[:8]} software-failed at entry: {failure}')
            secs = [r['seconds'] for f, r in screen_res.items() if f in screen]
            new[d] = dict(status='active' if pool else 'software-failed', wave=spec.get('wave', 0), entry_update=k,
                          screen_seconds=statistics.mean(secs) if secs else 60., teacher_pool=pool,
                          monitor_shard=shard if mon else None, mcts_dir=str(ref), batches=[], evals=[], reasons=[],
                          attempted=0, incomplete=0, zeroshot=None)
            if failure:
                new[d].update(status='software-failed', encode_failed=True)
                new[d]['reasons'].append(f'k={k}: software-failed: {failure}')
            elif not pool:
                new[d]['reasons'].append(f'k={k}: no completed teacher fights')
        self.state['decks'].update(new)
        mapping = self.split_manifest()
        final = [r for d in self.entered(failed=True) if not self.state['decks'][d].get('encode_failed') for r in self.rows(d, 'final', self.a.final)]
        write_starts(final, self.out / 'final-reserved.parquet')  # reserved: never played by the controller
        if self.state['model'] and k > 0:
            live = [d for d in ids if new[d]['status'] != 'software-failed']
            for d, ev in self.monitor_eval(f'zeroshot-{k}', live).items():
                batch = dict(k=k, kind='zeroshot', attempted=self.a.monitor, completed=ev['n'], wins=ev['wins'], seconds=0.)
                self.state['decks'][d]['batches'].append(batch)
                self.state['decks'][d]['zeroshot'] = dict(ev, k=k)
        self.log(f'entered {len(ids)} decks at update {k}: {ids}')
        self.save(stage_name=None)
        return mapping

    def monitor_eval(self, name, decks):
        """learned2k (no exploration) on monitor seeds, paired against the cached MCTS reference of each deck."""
        if not decks:
            return {}
        rows = [r for d in decks for r in self.rows(d, 'monitor', self.a.monitor)]
        self.save(stage_name=name)
        write_starts(rows, self.out / f'{name}-starts.parquet')
        model = Path(self.state['model'])
        run = play(self.out, name, self.out / f'{name}-starts.parquet', model, self.worker, self.a.sims, self.a.workers)
        result = {}
        for d in decks:
            starts = [{'fight_id': f} for f in self.monitor_ids(d)]
            try:
                s = score(starts, Path(self.state['decks'][d]['mcts_dir']), run)
                result[d] = dict(n=s['n'], wins=s['wins'], mcts_wins=round(s['mcts_win_rate'] * s['n']),
                                 gap=s['gap'], gap_se=s['gap_se'])
            except ValueError:
                result[d] = dict(n=0, wins=0, mcts_wins=0, gap=None, gap_se=None)
        return result

    def train(self, k, ids_by_deck, bootstrap=False, ctl=None):
        """Mixed-deck train.py run; `ids_by_deck` {deck: {fight_id: shard}}. Returns the update's train record."""
        ids = {f: s for m in ids_by_deck.values() for f, s in m.items()}
        groups = {f: d for d, m in ids_by_deck.items() for f in m}
        valid = [self.state['decks'][d]['monitor_shard'] for d in self.entered() if self.state['decks'][d]['monitor_shard']]
        shards = sorted(set(ids.values()) | set(valid))  # shards with no selected fight and no monitor rows are pruned
        udir = self.out / 'models' / str(k)
        udir.mkdir(parents=True, exist_ok=True)
        jwrite(udir / 'train-fights.json', sorted(ids))
        jwrite(udir / 'groups.json', groups)
        split = self.split_manifest()
        c = BOOTSTRAP if bootstrap else ctl
        cmd = ['/usr/bin/time', '-v', '-o', udir / 'time.txt', PY, '-m', 'agents.combat.pv.train', '--data', *shards,
               '--states-per-fight', STATES_PER_FIGHT, '--flat-policy-weighting', '--grad-clip', 1, '--epochs', c['epochs'],
               '--lr', c['lr'], '--mix-shards', '--groups', udir / 'groups.json', '--split-manifest', split,
               '--train-fights', udir / 'train-fights.json', '--device', self.a.device, '--out', udir / 'model']
        if bootstrap and not self.a.init:
            cmd += ['--value-activation', 'sigmoid']
        else:
            cmd += ['--init', Path(self.state['model']) / 'model.pt', '--resume-optimizer']
        self.save(stage_name=f'train-{k}')
        stage(self.out, f'train-{k}', cmd)
        self.state['model'], self.state['model_k'] = str(udir / 'model'), k
        rss = re.search(r'Maximum resident set size \(kbytes\): (\d+)', (udir / 'time.txt').read_text())
        epochs = [json.loads(l) for l in (self.out / f'logs/train-{k}.log').read_text().splitlines() if l.startswith('{"epoch"')]
        last = epochs[-1] if epochs else {}
        return dict(max_rss_mb=int(rss.group(1)) // 1024 if rss else None, fights=len(ids), shards=len(shards),
                    train=last.get('train'), val=last.get('val'), groups=(last.get('train') or {}).get('groups'))

    def evaluate(self, k):
        decks = [d for d in self.entered()]
        results = self.monitor_eval(f'eval-{k}', decks)
        for d, ev in results.items():
            rec = self.state['decks'][d]
            rec['evals'] = [e for e in rec['evals'] if e['k'] != k] + [dict(ev, k=k, p=smoothed_p(rec['batches']) if rec['batches'] else None)]
        return results

    def lifecycle(self, k, evaluated):
        for d in self.entered():
            rec = self.state['decks'][d]
            learner = sum(b['completed'] for b in rec['batches'] if b['kind'] == 'learner')
            p = smoothed_p(rec['batches']) if rec['batches'] else 0.5
            ev = evaluated.get(d)
            if ev and ev['n'] and ev['wins'] >= ev['mcts_wins'] and rec.get('mastered_at') is None:
                rec['mastered_at'] = k
                self.log(f'{d[:8]} learner matched MCTS on monitor at k={k} ({ev["wins"]} vs {ev["mcts_wins"]}); teacher taper starts')
            new, why = next_status(rec['status'], ev, p, learner, self.threshold)
            if new != rec['status']:
                rec['reasons'].append(f'k={k}: {rec["status"]} -> {new}: {why}')
                self.log(f'{d[:8]} {rec["status"]} -> {new}: {why}')
                rec['status'] = new

    def collect(self, k, plan):
        quotas = plan['quotas']
        rows = [r for d in quotas for r in self.rows(d, f'learner-{k}', quotas[d])]
        write_starts(rows, self.out / f'collect-{k}-starts.parquet')
        self.save(stage_name=f'collect-{k}')
        run = play(self.out, f'collect-{k}', self.out / f'collect-{k}-starts.parquet', Path(self.state['model']),
                   self.worker, self.a.sims, self.a.workers, explore=True)
        shard = str(encode(self.out, f'rows-{k}', run, self.worker))
        results = read_results(run)
        for d, q in quotas.items():
            rec = self.state['decks'][d]
            res = [results.get(f'{self.ns}:{d}:learner-{k}:{i}') for i in range(q)]
            done = [(f'{self.ns}:{d}:learner-{k}:{i}', r) for i, r in enumerate(res) if r and r['status'] == 'completed']
            batch = dict(k=k, kind='learner', attempted=q, completed=len(done), wins=sum(r['fight']['won'] for _, r in done),
                         seconds=sum(r['seconds'] for _, r in done), ids=[f for f, _ in done], shard=shard)
            rec['batches'] = [b for b in rec['batches'] if not (b['kind'] == 'learner' and b['k'] == k)] + [batch]
            rec['attempted'] = sum(b['attempted'] for b in rec['batches'] if b['kind'] == 'learner')
            rec['incomplete'] = rec['attempted'] - sum(b['completed'] for b in rec['batches'] if b['kind'] == 'learner')
            if q and (q - len(done)) / q > MAX_INCOMPLETE and rec['status'] != 'software-failed':
                rec['reasons'].append(f'k={k}: software-failed: {q - len(done)}/{q} incomplete')
                self.log(f'{d[:8]} software-failed: {q - len(done)}/{q} incomplete in update {k}')
                rec['status'] = 'software-failed'

    def make_plan(self, k):
        ctl = self.load_control()
        enter = []
        if k == 0:
            enter = [d for d in self.order if self.decks[d].get('wave', 0) == 0]
        elif k % ctl['wave_every'] == 0:
            later = sorted((d for d in self.order if d not in self.state['decks'] and self.decks[d].get('wave', 0) > 0),
                           key=lambda d: (self.decks[d]['wave'], self.order.index(d)))
            enter = later[:ctl['wave_size']]
        return dict(k=k, ctl=ctl, enter=enter, quotas=None)

    def quotas(self, k, plan):
        ctl = plan['ctl']
        rows, table = [], []
        for d in self.entered():
            rec = self.state['decks'][d]
            if rec['status'] not in ('active', 'graduated'):
                continue
            p, secs = smoothed_p(rec['batches']), fight_seconds(rec)
            weight = max(0.05, p * (1 - p)) / secs
            rows.append((d, rec['status'] == 'graduated', weight))
            table.append(dict(deck=d, status=rec['status'], p=p, seconds=secs, weight=weight))
        quotas = allocate(ctl['fights_per_update'], ctl['min_fights_per_deck'], rows)
        for row in table:
            row['quota'] = quotas[row['deck']]
        self.state['allocations'][str(k)] = table
        self.log(f'allocation k={k}:\n' + '\n'.join(
            f'  {r["deck"][:8]} {r["status"]:9} p={r["p"]:.3f} sec={r["seconds"]:.1f} w={r["weight"]:.5f} quota={r["quota"]}' for r in table))
        return quotas

    def update(self, k):
        plan = self.state['plan']
        if not plan or plan['k'] != k:
            plan = self.state['plan'] = self.make_plan(k)
            self.save()
        ctl = plan['ctl']
        fresh = [d for d in plan['enter'] if d not in self.state['decks']]
        if fresh:
            self.entry(k, fresh, ctl)
        record = dict(k=k, started=time.time())
        evaluated = {}
        if k == 0:
            if not self.a.init:
                wave = {d: dict(self.state['decks'][d]['teacher_pool']) for d in self.entered()}
                record['train'] = self.train(0, wave, bootstrap=True)
            else:
                self.state['model'], self.state['model_k'] = str(self.a.init), 0
            evaluated = self.evaluate(0)
        else:
            if plan['quotas'] is None:
                plan['quotas'] = self.quotas(k, plan)
                self.save()
            self.collect(k, plan)
            self.save()
            chosen = {d: training_ids(self.state['decks'][d], k, ctl) for d in self.entered()}
            record['train'] = self.train(k, chosen, ctl=ctl)
            if k % ctl['eval_every'] == 0:
                evaluated = self.evaluate(k)
        self.lifecycle(k, evaluated)
        record['evaluated'] = bool(evaluated)
        record['finished'] = time.time()
        self.state['updates'][str(k)] = record
        self.state['k'], self.state['plan'] = k, None
        self.save(phase='idle')
        if evaluated:
            self.reports()
        self.log(f'update {k} done: {json.dumps({x: record.get(x) for x in ("train",)})[:300]}')

    def reports(self):
        lines = ['update,deck,status,collection_p,monitor_wins,mcts_wins,monitor_n']
        for d in self.entered(failed=True):
            rec = self.state['decks'][d]
            for e in rec['evals']:
                lines.append(f'{e["k"]},{d},{rec["status"]},{"" if e["p"] is None else round(e["p"], 4)},{e["wins"]},{e["mcts_wins"]},{e["n"]}')
        (self.out / 'curves.csv').write_text('\n'.join(lines) + '\n')
        recs = [self.state['decks'][d] for d in self.entered()]
        agg = macro(recs)
        counts = {s: sum(1 for d in self.state['decks'].values() if d['status'] == s)
                  for s in ('active', 'graduated', 'parked', 'software-failed')}
        md = [f'# Human-deck corpus controller (update {self.state["k"]})', '',
              f'Denominators: entered {len(self.state["decks"])}, not yet entered {len(self.decks) - len(self.state["decks"])}, {counts}', '']
        if agg:
            se = 'n/a' if agg['gap_se'] is None else f'{100 * agg["gap_se"]:.1f}'
            md += [f'Macro over {agg["decks"]} evaluated decks (latest monitor): learned2k {100 * agg["learned"]:.1f}% vs MCTS20k '
                   f'{100 * agg["mcts"]:.1f}%; paired gap {100 * agg["gap"]:+.1f} ± {se} pts (SE clustered by deck; monitor seeds are '
                   f'selection-consumed, not a final test).', '']
        md += ['| deck | status | entry | zero-shot | monitor (latest) | MCTS | learner fights (incomplete) |', '|---|---|---:|---:|---:|---:|---:|']
        for d in self.entered(failed=True):
            rec, e = self.state['decks'][d], (self.state['decks'][d]['evals'] or [None])[-1]
            z = rec['zeroshot']
            md.append(f'| {d[:8]} | {rec["status"]} | {rec["entry_update"]} | {"-" if not z else f"{z["wins"]}/{z["n"]}"} | '
                      f'{"-" if not e else f"{e["wins"]}/{e["n"]} (k={e["k"]})"} | {"-" if not e else e["mcts_wins"]} | '
                      f'{rec["attempted"]} ({rec["incomplete"]}) |')
        (self.out / 'REPORT.md').write_text('\n'.join(md) + '\n')

    def run(self):
        while True:
            k = self.state['k'] + 1
            if k > self.a.updates:
                self.log(f'reached --updates {self.a.updates}')
                return
            if self.state['plan'] is None and self.load_control()['stop']:
                self.log('stop requested in control.json; exiting cleanly')
                self.save(phase='stopped')
                return
            self.save(phase=f'update-{k}')
            self.update(k)


def freeze(a):
    out = a.run.resolve()
    state = json.loads((out / 'state.json').read_text())
    k = state['model_k']
    dest = out / 'frozen-models' / str(k)
    if not dest.exists():
        shutil.copytree(state['model'], dest)
    grad = [dict(deck_id=d, entry_update=r['entry_update'], wave=r['wave'],
                 latest_eval=(r['evals'] or [None])[-1], reasons=r['reasons'])
            for d, r in state['decks'].items() if r['status'] == 'graduated']
    jwrite(dest / 'graduated.json', dict(update=k, model=str(dest), graduated=grad))
    print(f'froze update {k} -> {dest}; graduated {len(grad)}: {[g["deck_id"][:8] for g in grad]}')


def status(a):
    out = a.run.resolve()
    sp = out / 'state.json'
    if not sp.exists():
        print('controller not initialized'); return
    s = json.loads(sp.read_text())
    ctl = {**DEFAULTS, **(json.loads((out / 'control.json').read_text()) if (out / 'control.json').exists() else {})}
    print(f'CORPUS {out.name}: phase {s["phase"]} stage {s["stage"]} update k={s["k"]} elapsed {(time.time() - s["started"]) / 3600:.2f} h '
          f'(state age {time.time() - s.get("updated", s["started"]):.0f}s) model_k={s["model_k"]} stop={ctl["stop"]}')
    counts = {}
    for r in s['decks'].values():
        counts[r['status']] = counts.get(r['status'], 0) + 1
    print(f'decks entered {len(s["decks"])} {counts}; control {ctl}')
    alloc = {r['deck']: r for r in s['allocations'].get(str(s['k'] + (1 if s['plan'] else 0)), s['allocations'].get(str(s['k']), []))}
    for d, r in s['decks'].items():
        e = (r['evals'] or [None])[-1]
        q = alloc.get(d)
        print(f'{d[:8]} {r["status"]:13} entry {r["entry_update"]:3} p={smoothed_p(r["batches"]):.2f} '
              f'quota {"-" if not q else q["quota"]:>3} monitor {"-" if not e else f"{e["wins"]}/{e["n"]}@{e["k"]}"} '
              f'mcts {"-" if not e else e["mcts_wins"]} learner {r["attempted"]} inc {r["incomplete"]}')
    u = s['updates'].get(str(s['k']))
    if u and u.get('train'):
        t = u['train']
        print(f'last train (update {s["k"]}): max RSS {t["max_rss_mb"]} MB, {t["fights"]} fights/{t["shards"]} shards, '
              f'train {json.dumps(t["train"] and {x: round(v, 4) for x, v in t["train"].items() if x.endswith("loss")})} '
              f'val {json.dumps(t["val"] and {x: round(v, 4) for x, v in t["val"].items() if x.endswith("loss")})}')
    print(f'logs: {out}/logs/controller.log, {out}/logs/<stage>.log; report {out}/REPORT.md; curves {out}/curves.csv')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--decks', type=Path, required=True)
    r.add_argument('--worker', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--workers', type=int, default=10)
    r.add_argument('--namespace', default='hdc-v1')
    r.add_argument('--init', type=Path, help='model dir (model.pt/model.onnx); omit for a fresh bootstrap')
    r.add_argument('--device', default='cuda')
    r.add_argument('--sims', type=int, default=2000)
    r.add_argument('--teacher-sims', type=int, default=20000)
    r.add_argument('--monitor', type=int, default=30)
    r.add_argument('--final', type=int, default=30)
    r.add_argument('--updates', type=int, default=10**6, help='stop after this update index (may be raised on resume)')
    r.add_argument('--exclude-starts', type=Path, nargs='*', default=[p for p in EXCLUDE_DEFAULT if p.exists()],
                   help='starts parquets whose seeds are never reused (pool, bench, library, screen probe/fill)')
    for name, fn in (('status', status), ('freeze', freeze)):
        s = sub.add_parser(name)
        s.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'status':
        return status(a)
    if a.command == 'freeze':
        return freeze(a)
    if not 1 <= a.workers <= 10 or min(a.sims, a.teacher_sims, a.monitor, a.final) < 1:
        p.error('1–10 workers and positive budgets required')
    Controller(a).run()


if __name__ == '__main__':
    main()
