#!/usr/bin/env python3
"""Overnight continuation runner (OVERNIGHT.md). Run from repo root:  run.py {prep|smoke|run|report}
Resumable stage driver. Every game: dispatch-intent ledger row BEFORE start, result row on completion (fsynced).
Completed games are reused on resume; intent-without-result = incomplete attempt, never silently replayed.
Completed/orphan/errored gameplay is NEVER redispatched (no override exists); only UNSTARTED jobs run on resume.
The raw trainable line is stored inside the result row, so a result row is atomic (partial last line = orphan).
Exit codes: 0 finished (incl. deadline stop, reported), 2 Halt (SRE/research review, no auto restart), 3 unexpected crash (supervisor may resume after `verify`)."""
import argparse, calendar, concurrent.futures as cf, json, os, re, shutil, signal, statistics, subprocess, sys, threading, time, traceback, hashlib
from pathlib import Path
import pyarrow.parquet as pq

REPO = Path.cwd(); assert (REPO / 'AGENTS.md').exists(), 'run from repo root'
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'experiments/combat-agent-rescue/counterfactual-rescue/phase5'))
import ei  # noqa: E402  (phase5 runner: helpers, constants; main() is guarded)
from apps.run_rl.single_deck import select_loadout, generate  # noqa: E402

NS = "overnight-rollout-v1"
RUN = ei.D6 / f'id={NS}'; OUT = RUN / 'out'; LOGS = RUN / 'logs'
HERE = Path(__file__).resolve().parent
P5 = ei.OUT                                   # phase5 out dir
CHAMP = P5 / 'strong/update3/model'           # exact Phase5 strong-u3 (weights + AdamW moments)
REPS = {'A': 1, 'B': 2}                        # replica -> trainer rng seed
UPD, NTRAIN, NMON, NFIN, MIN_DONE, WINDOW = 20, 300, 200, 600, 270, 6
WORKERS, TIMEOUT, MIX = 10, 900, '0.5'
MON_AT = (5, 10, 15, 20)
TZ = '+0100'
utc = lambda s: calendar.timegm(time.strptime(s, '%Y-%m-%d %H:%M:%S'))
DEADLINE = utc('2026-10-07 06:15:00')        # = 07:15 BST
NO_NEW_UPDATE = utc('2026-10-07 05:15:00')   # = 06:15 BST
BATCH = 100
MEM_MIN_GB, MEM_SECS = 1.2, 60
now, log, sha = ei.now, ei.log, ei.sha
LOCK = threading.Lock(); ACTIVE = ei.ACTIVE; HALT = threading.Event(); DONE = threading.Event()
PHASE6_DIR = ei.D6 / 'id=demon-form-rollout-confirmation-v1'


class Halt(Exception): pass


def kill_active():
    for _ in range(5):
        try:
            kill_active()
            return
        except RuntimeError: time.sleep(0.05)   # set mutated by worker threads


def wjson(p, o):
    t = Path(str(p) + '.tmp'); t.write_text(json.dumps(o, indent=1, default=str)); os.replace(t, p)


def frozen(n): return OUT / 'frozen' / n


# ------------------------------------------------------------------ ledger
class Ledger:
    """Append-only. Validates on load: one intent and at most one result per key, result needs intent. No overwrite, no redispatch."""
    def __init__(self):
        self.path = OUT / 'ledger.jsonl'; self.res = {}; self.intent = {}; self.partial_tail = False
        if self.path.exists():
            raw = self.path.read_text(); ls = raw.split('\n')
            if ls and ls[-1] == '': ls.pop()
            elif ls: self.partial_tail = True                       # last line unterminated
            for n, l in enumerate(ls):
                try: r = json.loads(l)
                except json.JSONDecodeError:
                    if n == len(ls) - 1 and self.partial_tail: log('LEDGER: unterminated final row (crash mid-write) set aside'); self.partial_tail = True; continue
                    raise Halt(f'ledger corrupt at line {n+1}')
                if r['kind'] == 'intent':
                    if r['key'] in self.intent: raise Halt(f'ledger: duplicate intent {r["key"]}')
                    self.intent[r['key']] = r
                elif r['kind'] == 'result':
                    if r['key'] in self.res: raise Halt(f'ledger: duplicate result {r["key"]}')
                    if r['key'] not in self.intent: raise Halt(f'ledger: result without intent {r["key"]}')
                    self.res[r['key']] = r
                else: raise Halt(f'ledger: unknown row kind at line {n+1}')
        if self.partial_tail:      # preserve the fragment as provenance, then cut it so the ledger stays line-valid
            keep = raw[:raw.rfind('\n') + 1]; frag = raw[len(keep):]
            (OUT / f'ledger.partial-tail.{int(time.time())}.txt').write_text(frag); self.path.write_text(keep); self.partial_tail = False
        self.f = open(self.path, 'a')
    @property
    def lines(self): return {k: r['line'] for k, r in self.res.items() if r.get('line')}

    def write(self, rec):
        with LOCK:
            if rec['kind'] == 'intent':
                if rec['key'] in self.intent: raise Halt(f'refusing second dispatch of {rec["key"]}')
                self.intent[rec['key']] = rec
            else:
                if rec['key'] in self.res or rec['key'] not in self.intent: raise Halt(f'bad result write {rec["key"]}')
                self.res[rec['key']] = rec
            self.f.write(json.dumps(rec) + '\n'); self.f.flush(); os.fsync(self.f.fileno())

    def state(self, key, need_line=False):
        r = self.res.get(key)
        if r is None: return 'orphan' if key in self.intent else 'new'
        if r['status'] == 'completed' and need_line and not r.get('line'): return 'orphan'   # completed but no raw line: incomplete
        return r['status']


def play(cmd, key, fight_id, start, L, meta):
    """One game in its own process group. Intent row first; returns result record (and trainable line)."""
    if HALT.is_set(): return None, None
    L.write({'kind': 'intent', 'key': key, 'at': now(), 'cmd': [Path(cmd[0]).name] + cmd[1:], **meta})
    t0 = time.time(); p = None; err = None; out = se = ''
    try:
        with LOCK:
            if HALT.is_set(): return None, None
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True); ACTIVE.add(p)
        out, se = p.communicate(json.dumps({'fight_id': fight_id, 'start': start}) + '\n', timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        ei.kill(p); err = f'{TIMEOUT}s timeout'
    except BaseException as e:
        ei.kill(p); err = f'{type(e).__name__}: {e}'
    finally:
        if p:
            with LOCK: ACTIVE.discard(p)
    rec = {'kind': 'result', 'key': key, 'fight_id': fight_id, 'started': t0, 'seconds': time.time() - t0, 'error': err, 'logged': now(),
           'returncode': None if err else p.returncode, 'stderr': (se or '')[-1000:], **meta}
    try: r = json.loads(out) if not err and p.returncode == 0 and out.count('\n') == 1 else None
    except json.JSONDecodeError: r = None
    if not r or r.get('status') not in ('completed', 'capped'): rec['status'] = 'error'; return rec, None
    rec['status'] = r['status']
    agent = (r.get('fight') or {}).get('agent') or (r['search'][0]['agent'] if r.get('search') else '')
    rec['agent'] = agent
    want = ('mcts' in agent and 'sims=20000' in agent) if cmd[1] == 'teacher' else (('rollout_mix=0.500000' in agent) == (MIX in cmd))
    if not want: rec.update(status='error', error=f'agent mismatch {agent}'); return rec, None
    if r['status'] == 'completed':
        f = r['fight']; rec.update(won=bool(f['won']), hp=f['final_hp'], actions=len(f['actions']))
        line = {'fight_id': fight_id, 'start': f['start'], 'actions': f['actions'], 'won': f['won'], 'final_hp': f['final_hp'],
                'search': [{'step': s['step'], 'root_value': s['root_value'], 'children': s['children']} for s in r.get('search', [])]}
        return rec, line
    return rec, None


def pv_cmd(model): return [str(ei.worker_bin()), 'play', str(Path(model) / 'model.onnx'), '2000', '--rollout-mix', MIX]
def teacher_cmd(): return [str(ei.worker_bin()), 'teacher', '20000']


def dispatch(L, stage, jobs, keep_lines=False):
    """jobs: (key, cmd, fight_id, start, meta). Reuses completed results; orphans (and completed-without-line when keep_lines) are counted, never replayed;
    prior error halts; only 'new' keys are started. First error halts everything and kills running games (their intents remain orphans)."""
    results = {}; todo = []; orphans = []
    for key, cmd, fid, st, meta in jobs:
        s = L.state(key, keep_lines)
        if s in ('completed', 'capped'): results[key] = L.res[key]
        elif s == 'error': raise Halt(f'{stage}: prior error on {key}; no auto retry, needs SRE/Astra: {L.res[key].get("error")} {L.res[key]["stderr"][-200:]}')
        elif s == 'orphan': orphans.append(key)
        else: todo.append((key, cmd, fid, st, meta))
    if orphans: log(stage, 'ORPHAN/incomplete attempts (not replayed):', len(orphans))
    t0 = time.time(); n = 0; pool = cf.ThreadPoolExecutor(WORKERS)
    try:
        futs = {pool.submit(play, cmd, key, fid, st, L, {**meta, 'stage': stage}): key for key, cmd, fid, st, meta in todo}
        for fu in cf.as_completed(futs):
            rec, line = fu.result()
            if rec is None: continue
            if keep_lines and rec['status'] == 'completed': rec['line'] = line
            L.write(rec); results[rec['key']] = rec; n += 1
            if rec['status'] == 'error':
                HALT.set(); raise Halt(f'{stage}: error {rec["key"]} {rec.get("error")} {rec["stderr"][-300:]}')
            if n % 100 == 0: log(stage, f'{n}/{len(todo)} new games ({time.time()-t0:.0f}s)')
    except BaseException:
        HALT.set(); pool.shutdown(wait=False, cancel_futures=True); kill_active(); pool.shutdown(wait=True); raise
    pool.shutdown(wait=True)
    return results, orphans


# ------------------------------------------------------------------ memory guard
def mem_guard():
    low = 0
    while not (HALT.is_set() or DONE.is_set()):
        time.sleep(5)
        avail = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable')) / 1e6
        low = low + 5 if avail < MEM_MIN_GB else 0
        if low >= MEM_SECS:
            log('MEMORY PRESSURE', avail, 'GB available: halting dispatch'); HALT.set()
            for p in list(ACTIVE): ei.kill(p)
            return


# ------------------------------------------------------------------ prep: frozen files, seeds, manifest
def prep():
    if (OUT / 'manifest.json').exists(): raise SystemExit('already prepared (immutable)')
    if sha(ei.WORKER) != ei.WORKER_SHA: raise SystemExit('worker sha')
    (OUT / 'frozen').mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    for src, n in ((ei.WORKER, 'pv_worker'), (HERE / 'train_fixed.py', 'train_fixed.py'), (HERE / 'run.py', 'run.py'), (ei.__file__, 'ei.py'),
                   (HERE.parent / 'OVERNIGHT.md', 'OVERNIGHT.md')): shutil.copy2(src, frozen(n))
    used, files, cfg = ei.used_seeds(); n0 = len(used)
    for u in (1, 2, 3): used |= {r['start']['seed'] for r in json.loads((P5 / f'starts-u{u}.json').read_text())}
    n1 = len(used)
    base = select_loadout(pq.read_table(cfg['source']).to_pylist(), cfg['deck_id'])
    if (base['start']['hp'], base['start']['max_hp'], base['start']['potions'], cfg['deck_id']) != (34, 52, [1, 1], '249e5246-153d-4030-bc11-598774146147'): raise SystemExit('loadout')
    dev0 = pq.read_table(ei.DEV / 'starts.parquet').to_pylist()[0]['start']
    starts = {}
    for r in REPS:
        for u in range(1, UPD + 1): starts[f'train-{r}-u{u:02d}'] = generate(base, NS, f'train-{r}-u{u:02d}', NTRAIN, used)
    starts['monitor'] = generate(base, NS, 'monitor', NMON, used); starts['final'] = generate(base, NS, 'final', NFIN, used)
    starts['smoke'] = generate(base, NS, 'smoke', 4, used)
    allseeds = [x['start']['seed'] for v in starts.values() for x in v]
    if len(set(allseeds)) != len(allseeds) or len(used) != n1 + len(allseeds): raise SystemExit('seed collision')
    strip = lambda s: {k: v for k, v in s.items() if k not in ('seed', 'misc_rng', 'potion_rng')}
    if any(strip(v[0]['start']) != strip(dev0) for v in starts.values()): raise SystemExit('start differs from dev start beyond seed fields')
    wjson(OUT / 'starts.json', starts)
    champ = {n: sha(CHAMP / n) for n in ('model.pt', 'model.onnx', 'model.onnx.data')}
    old = ei.OLD_SHARDS + [P5 / f'strong/update{u}/rows.parquet' for u in (1, 2, 3)]
    man = {'protocol_sha256': sha(HERE.parent / 'OVERNIGHT.md'), 'frozen_sha256': {n: sha(frozen(n)) for n in ('pv_worker', 'train_fixed.py', 'run.py', 'ei.py')},
           'champion_sha256': champ, 'old_shards_sha256': {str(p): sha(p) for p in old}, 'starts_sha256': sha(OUT / 'starts.json'),
           'counts': {k: len(v) for k, v in starts.items() if not k.startswith('train-')} | {'train_total': sum(len(v) for k, v in starts.items() if k.startswith('train-'))},
           'excluded_seed_files': files, 'excluded_seeds_prior': n0, 'excluded_seeds_incl_phase5': n1, 'namespace': NS, 'rng_seeds': REPS, 'frozen_at': now()}
    wjson(OUT / 'manifest.json', man); log('PREP DONE', man['counts'], 'excluded', n1)


def load_starts(): return json.loads((OUT / 'starts.json').read_text())
MODEL_FILES = ('model.pt', 'model.onnx', 'model.onnx.data')


def check_frozen(warn_protocol=True):
    """Verify actual executing sources (run.py, imported ei.py), frozen copies, seeds, champion and old shards against the CURRENT manifest."""
    man = json.loads((OUT / 'manifest.json').read_text()); fz = man['frozen_sha256']
    for n, h in fz.items():
        if sha(frozen(n)) != h: raise Halt(f'frozen file changed: {n}')
    if sha(Path(__file__)) != fz['run.py']: raise Halt('executing run.py differs from manifest (use `refreeze` with a reason BEFORE games, or recorded amendment)')
    if sha(ei.__file__) != fz['ei.py']: raise Halt('imported ei.py differs from manifest')
    if sha(OUT / 'starts.json') != man['starts_sha256']: raise Halt('starts changed')
    if {n: sha(CHAMP / n) for n in MODEL_FILES} != man['champion_sha256']: raise Halt('champion changed')
    for p, h in man['old_shards_sha256'].items():
        if sha(p) != h: raise Halt(f'old shard changed {p}')
    if warn_protocol and sha(HERE.parent / 'OVERNIGHT.md') != man['protocol_sha256']: log('WARNING: OVERNIGHT.md differs from manifest protocol hash')
    ei.OUT = OUT   # worker_bin() now resolves OUT/frozen/pv_worker
    if ei.worker_bin() != frozen('pv_worker') or sha(frozen('pv_worker')) != ei.WORKER_SHA: raise Halt('worker not the frozen copy')
    return man


def phase6_disjoint(starts):
    """Assert overnight seeds are disjoint from everything in the Phase6 run dir (actual 600 once recorded) and the reserved final file."""
    mine = {str(x['start']['seed']) for v in starts.values() for x in v}; hits = set(); scanned = 0
    files = [p for p in PHASE6_DIR.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.parquet')] if PHASE6_DIR.exists() else []
    for p in files + [ei.ORIG / 'final-reserved.parquet']:
        scanned += 1
        text = str(pq.read_table(p).to_pylist()) if p.suffix == '.parquet' else p.read_text(errors='ignore')
        hits |= set(re.findall(r'\d{9,20}', text)) & mine
    if hits: raise Halt(f'overnight seeds overlap Phase6/reserved: {sorted(hits)[:3]}')
    return scanned


# ------------------------------------------------------------------ training
def valid_model(d, rec_hashes=None):
    """A checkpoint counts only with train.json + checkpoint + ONNX + external data whose hashes match train.json (and state record if given)."""
    if not (d / 'train.json').exists() or any(not (d / n).exists() for n in MODEL_FILES): return False
    h = json.loads((d / 'train.json').read_text())['files_sha256']
    now_h = {n: sha(d / n) for n in MODEL_FILES}
    if any(h.get(n) != now_h[n] for n in MODEL_FILES): return False
    return rec_hashes is None or all(rec_hashes.get(n) == now_h[n] for n in MODEL_FILES)


def train(rep, u, init, new_files, n_new, tag):
    d = OUT / f'rep{rep}/u{u:02d}'
    if valid_model(d / 'model'): return d / 'model'      # intact earlier result: reuse (same frozen data/RNG)
    if (d / 'model').exists(): raise Halt(f'{tag}: model/ exists but fails marker/hash validation; ambiguous, not deleting (needs review)')
    shutil.rmtree(d / 'model.tmp', ignore_errors=True)    # only an unambiguous partial write
    old = ei.OLD_SHARDS + [P5 / f'strong/update{k}/rows.parquet' for k in (1, 2, 3)]
    ids = []
    for f in old + new_files: ids += list(set(pq.read_table(f, columns=['fight_id'])['fight_id'].to_pylist()))
    if len(ids) != len(set(ids)): raise Halt('duplicate fight ids across old/new pools')
    (d / 'split.json').write_text(json.dumps({i: 'train' for i in ids}))
    cmd = [str(ei.PY), str(frozen('train_fixed.py')), '--old', *map(str, old), '--new', *map(str, new_files), '--split', str(d / 'split.json'), '--init', str(init / 'model.pt'),
           '--out', str(d / 'model.tmp'), '--update', str(u), '--rng-seed', str(REPS[rep]), '--expect-old-fights', '2032', '--expect-new-fights', str(n_new)]
    log('TRAIN start', tag, 'new_fights', n_new); t0 = time.time()
    with open(d / 'train.log', 'w') as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        with LOCK: ACTIVE.add(p)
        p.wait()
        with LOCK: ACTIVE.discard(p)
    if p.returncode: raise Halt(f'train {tag} rc={p.returncode}: {(d / "train.log").read_text()[-600:]}')
    rp = json.loads((d / 'model.tmp/train.json').read_text())
    if rp['counts'] != {'old': 32000, 'new': 32000, 'steps': 1000} or rp['old_fights'] != 2032 or rp['test_override_steps'] or rp['rng_seed'] != REPS[rep]: raise Halt(f'train contract {tag}')
    if rp['init_sha256'] != sha(init / 'model.pt'): raise Halt(f'train init mismatch {tag}')
    os.rename(d / 'model.tmp', d / 'model'); log('TRAIN done', tag, f'{time.time()-t0:.0f}s'); return d / 'model'


# ------------------------------------------------------------------ main driver
def wins(res, keys):
    """(wins, completed) over keys that have a completed result; missing/orphan/error/capped contribute neither a win nor a completion."""
    done = [k for k in keys if k in res and res[k]['status'] == 'completed']
    return sum(bool(res[k].get('won')) for k in done), len(done)


def stage_eval(L, st, name, cmd, starts, extra=None):
    jobs = [(f'{name}|{x["fight_id"]}', cmd, x['fight_id'], x['start'], {'cell': name, **(extra or {})}) for x in starts]
    res, orph = dispatch(L, name, jobs)
    keys = [j[0] for j in jobs]; w, c = wins(res, keys)
    return {'wins': w, 'completed': c, 'intended': len(jobs), 'orphans': len(orph), 'capped': sum(res[k]['status'] == 'capped' for k in keys if k in res),
            'per_start': {k.split('|', 1)[1]: bool(res[k].get('won')) for k in keys if k in res and res[k]['status'] == 'completed'},
            'seconds': [res[k]['seconds'] for k in keys if k in res], 'complete': c == len(jobs) and not orph}


def save(st): wjson(OUT / 'state.json', st)


def spg(L):  # measured mean seconds per game (recent completed)
    s = [r['seconds'] for r in L.res.values() if r['status'] == 'completed']; return statistics.mean(s[-400:]) if s else 8.0


def expected_model(r, u, st):
    """Model that MUST be used to collect update u: champion for u=1, else verified checkpoint u-1 of the same replica."""
    if u == 1: return CHAMP
    d = OUT / f'rep{r}/u{u-1:02d}/model'
    if not valid_model(d, st['train'].get(f'{r}-u{u-1}', {}).get('files_sha256')): raise Halt(f'previous checkpoint {r}-u{u-1} missing/invalid at resume')
    return d


def run():
    man = check_frozen(); starts = load_starts(); L = Ledger()
    phase6_disjoint(starts)
    st = json.loads((OUT / 'state.json').read_text()) if (OUT / 'state.json').exists() else {'created': now(), 'monitor': {}, 'collect': {}, 'train': {}, 'stopped': {}, 'alerts': []}
    if not (OUT / 'smoke-passed.json').exists(): raise Halt('smoke not passed')
    threading.Thread(target=mem_guard, daemon=True).start()
    st['runs'] = st.get('runs', []) + [{'start': now(), 'run_py_sha256': sha(Path(__file__)), 'ei_py_sha256': sha(ei.__file__), 'manifest_sha256': sha(OUT / 'manifest.json'),
                                         'manifest_attempt': man.get('attempt', 1), 'old_shards_verified': True}]; save(st)
    ref = st.setdefault('ref', {})
    if not ref.get('done'):                                       # reference monitors (once); both must be complete
        ref['champion'] = stage_eval(L, st, 'mon-ref-champion', pv_cmd(CHAMP), starts['monitor']); save(st)
        ref['teacher'] = stage_eval(L, st, 'mon-ref-teacher', teacher_cmd(), starts['monitor']); save(st)
        if not (ref['champion']['complete'] and ref['teacher']['complete']): raise Halt('reference monitor incomplete (orphan/cap); no replacement games')
        ref['done'] = True; save(st); log('REF monitor champion', ref['champion']['wins'], 'teacher', ref['teacher']['wins'])
    champ_w = ref['champion']['wins']
    models = {r: CHAMP for r in REPS}; low_streak = {r: 0 for r in REPS}
    for u in range(1, UPD + 1):
        t_u = time.time()
        active = [r for r in REPS if r not in st['stopped']]
        if not active: break
        est = st.get('update_seconds', 0) * 1.2
        if time.time() >= NO_NEW_UPDATE or time.time() + est > DEADLINE - 3600:
            st['deadline_stop'] = {'before_update': u, 'at': now(), 'est_update_s': est}; save(st); log('DEADLINE: no new update from', u); break
        jobs = []
        for r in active:
            d = OUT / f'rep{r}/u{u:02d}'
            if valid_model(d / 'model', st['train'].get(f'{r}-u{u}', {}).get('files_sha256')): models[r] = d / 'model'; continue
            mdl = expected_model(r, u, st)
            if models[r] != mdl: models[r] = mdl
            if sha(models[r] / 'model.pt') != (man['champion_sha256']['model.pt'] if u == 1 else st['train'][f'{r}-u{u-1}']['files_sha256']['model.pt']): raise Halt(f'collect model mismatch {r} u{u}')
            jobs += [(f'collect|{r}|{u}|{x["fight_id"]}', pv_cmd(models[r]), x['fight_id'], x['start'], {'rep': r, 'update': u, 'model_pt_sha': sha(models[r] / 'model.pt')})
                     for x in starts[f'train-{r}-u{u:02d}']]
        if jobs: dispatch(L, f'collect-u{u:02d}', jobs, keep_lines=True)
        for r in active:
            d = OUT / f'rep{r}/u{u:02d}'; d.mkdir(parents=True, exist_ok=True)
            if valid_model(d / 'model', st['train'].get(f'{r}-u{u}', {}).get('files_sha256')): continue
            keys = [f'collect|{r}|{u}|{x["fight_id"]}' for x in starts[f'train-{r}-u{u:02d}']]
            done = [k for k in keys if L.state(k, True) == 'completed']
            st['collect'][f'{r}-u{u}'] = {'intended': NTRAIN, 'completed': len(done), 'capped': sum(L.state(k) == 'capped' for k in keys),
                                          'orphans': sum(L.state(k, True) == 'orphan' for k in keys), 'wins': sum(bool(L.res[k]['won']) for k in done),
                                          'cpu_s': sum(L.res[k]['seconds'] for k in done)}
            if len(done) < MIN_DONE:
                st['stopped'][r] = f'u{u}: only {len(done)}/{NTRAIN} usable (<{MIN_DONE}); pending review'; st['alerts'].append(st['stopped'][r]); save(st); log('ALERT', r, st['stopped'][r]); continue
            if not (d / 'rows.parquet').exists(): ei.encode(sorted([L.res[k]['line'] for k in done], key=lambda l: l['fight_id']), d / 'rows.parquet')
        save(st)
        for r in [x for x in active if x not in st['stopped']]:        # sequential training
            d = OUT / f'rep{r}/u{u:02d}'
            if valid_model(d / 'model', st['train'].get(f'{r}-u{u}', {}).get('files_sha256')): models[r] = d / 'model'; continue
            lo = max(1, u - WINDOW + 1); files = [OUT / f'rep{r}/u{k:02d}/rows.parquet' for k in range(lo, u + 1)]
            n_new = sum(st['collect'][f'{r}-u{k}']['completed'] for k in range(lo, u + 1))
            if any(not f.exists() for f in files) or n_new > 1800: raise Halt(f'recent pool inconsistent {r} u{u}')
            m = train(r, u, models[r], files, n_new, f'{r}-u{u}'); models[r] = m
            rp = json.loads((m / 'train.json').read_text())
            st['train'][f'{r}-u{u}'] = {k: rp[k] for k in ('counts', 'old_fights', 'new_fights', 'new_states', 'train_seconds', 'files_sha256', 'onnx_max_abs_diff', 'init_sha256')} | {'final_log': rp['log'][-1]}; save(st)
        if u in MON_AT:
            for r in [x for x in REPS if x not in st['stopped']]:
                m = OUT / f'rep{r}/u{u:02d}/model'
                if not valid_model(m, st['train'].get(f'{r}-u{u}', {}).get('files_sha256')): raise Halt(f'monitor: checkpoint {r}-u{u} invalid')
                ev = st['monitor'][f'{r}-u{u}'] = stage_eval(L, st, f'mon|{r}|u{u}', pv_cmd(m), starts['monitor'], {'rep': r, 'update': u}); save(st)
                log('MONITOR', r, u, ev['wins'], 'champion', champ_w, 'complete', ev['complete'])
                low_streak[r] = low_streak[r] + 1 if (ev['wins'] <= champ_w - 20 and ev['complete']) else 0
                if low_streak[r] >= 2: st['stopped'][r] = f'regression rule at u{u} (>=20 below champion twice)'; st['alerts'].append(st['stopped'][r]); log('STOP replica', r, st['stopped'][r])
        st['update_seconds'] = time.time() - t_u; st['last_update_done'] = u; save(st); log('UPDATE DONE', u, f'{st["update_seconds"]:.0f}s', 'spg', f'{spg(L):.1f}')
    # selection (only complete scheduled evals)
    cands = [(v['wins'], -int(k.split('-u')[1]), 0 if k[0] == 'A' else 1, k) for k, v in st['monitor'].items() if v['complete']]
    sel = None
    if cands:
        best = max(cands, key=lambda c: (c[0], c[1], -c[2])); sel = {'key': best[3], 'monitor_wins': best[0], 'champion_monitor_wins': champ_w, 'promising': best[0] > champ_w}
    st['selection'] = sel; save(st); log('SELECTION', sel)
    if sel and sel['promising']: final(L, st, starts, sel)
    else: st['final'] = {'status': 'not run: no challenger strictly above champion monitor' if sel else 'not run: no complete monitor'}
    st['finished'] = now(); save(st); DONE.set(); report(st)


def final(L, st, starts, sel):
    rep, u = sel['key'].split('-u'); u = int(u); src = OUT / f'rep{rep}/u{u:02d}/model'; ch = OUT / 'challenger'
    if not valid_model(src, st['train'][sel['key']]['files_sha256']): raise Halt('selected checkpoint invalid')
    if not ch.exists():
        shutil.copytree(src, ch); st['challenger'] = {'source': str(src), 'sha256': {n: sha(ch / n) for n in MODEL_FILES}, 'snapshot_at': now()}; save(st)
    if st['challenger']['sha256'] != {n: sha(ch / n) for n in MODEL_FILES}: raise Halt('challenger snapshot changed')
    eta = NFIN * 3 * spg(L) / WORKERS * 1.15
    if time.time() + eta > DEADLINE - 900:
        st['final'] = {'status': f'not started: projected {eta/60:.0f} min does not fit before deadline-15min'}; save(st); log(st['final']); return
    jobs = []
    for b in range(-(-NFIN // BATCH)):
        for x in starts['final'][b * BATCH:(b + 1) * BATCH]:
            for arm, cmd in (('challenger', pv_cmd(ch)), ('champion', pv_cmd(CHAMP)), ('mcts20k', teacher_cmd())):
                jobs.append((f'final|{arm}|{x["fight_id"]}', cmd, x['fight_id'], x['start'], {'arm': arm, 'batch': b}))
    res, orph = dispatch(L, 'final', jobs)
    st['final'] = {'status': 'dispatched', 'orphans': len(orph), 'arms': {}}
    for arm in ('challenger', 'champion', 'mcts20k'):
        ks = [k for k in res if k.startswith(f'final|{arm}|')]; w, c = wins(res, ks); secs = [res[k]['seconds'] for k in ks]
        st['final']['arms'][arm] = {'completed': c, 'capped': sum(res[k]['status'] == 'capped' for k in ks), 'wins': w,
                                    'won': {k.split('|', 2)[2]: bool(res[k].get('won')) for k in ks if res[k]['status'] == 'completed'}, 'batch': {k.split('|', 2)[2]: res[k]['batch'] for k in ks},
                                    'sec_mean': statistics.mean(secs) if secs else None, 'sec_median': statistics.median(secs) if secs else None}
    st['final']['complete'] = all(a['completed'] == NFIN for a in st['final']['arms'].values()) and not orph; save(st)


def report(st=None):
    try: st = st or json.loads((OUT / 'state.json').read_text())
    except FileNotFoundError: st = {}
    o = ['# Overnight continuation report', f'generated {now()}', '']
    ref = st.get('ref', {}); g = lambda a: ref.get(a, {}).get('wins', 'n/a')
    o += [f'Monitor (200 starts) references: champion {g("champion")}, MCTS20k {g("teacher")}', '', '| ckpt | monitor wins | completed/intended | complete |', '|---|---|---|---|']
    for k, v in sorted(st.get('monitor', {}).items()): o.append(f'| {k} | {v["wins"]} | {v["completed"]}/{v["intended"]} | {v["complete"]} |')
    o += ['', f'Selection: {st.get("selection")}', f'Stopped replicas: {st.get("stopped")}', f'Deadline stop: {st.get("deadline_stop")}', f'Alerts: {st.get("alerts")}', '', '## Collection (wins/completed/capped per batch)']
    for k, v in st.get('collect', {}).items(): o.append(f'- {k}: {v["wins"]}/{v["completed"]} capped {v["capped"]} orphans {v["orphans"]} cpu_s {v["cpu_s"]:.0f}')
    f = st.get('final', {}); o += ['', '## Final', f'status: {f.get("status")}  complete: {f.get("complete")}']
    fa = f.get('arms') or {}
    for a, v in fa.items():
        fmt = lambda x: 'n/a' if x is None else f'{x:.1f}'
        o.append(f'- {a}: {v["wins"]}/{v["completed"]} of {NFIN} capped {v["capped"]} mean {fmt(v["sec_mean"])}s median {fmt(v["sec_median"])}s')
    for a, b in (('challenger', 'mcts20k'), ('challenger', 'champion'), ('champion', 'mcts20k')):
        if a in fa and b in fa and len(set(fa[a]['won']) & set(fa[b]['won'])) >= 2:
            n, d, se, ao, bo, p = ei.paired(fa[a]['won'], fa[b]['won']); o.append(f'- {a} - {b}: n={n} paired {100*d:+.1f}pp (approx 95% {100*(d-1.96*se):+.1f}..{100*(d+1.96*se):+.1f}) discordant {ao} vs {bo} exact McNemar p={p:.3f}')
    for b in range(-(-NFIN // BATCH)) if fa else []:
        o.append(f'  batch{b}: ' + ', '.join(f'{a} {sum(w for k, w in v["won"].items() if v["batch"].get(k) == b)}' for a, v in fa.items()))
    o += ['', f'challenger: {st.get("challenger")}', f'train seconds total: {sum(v.get("train_seconds", 0) for v in st.get("train", {}).values()):.0f}', f'runs: {st.get("runs")}',
          f'reproduce: .venv/bin/python experiments/combat-agent-rescue/counterfactual-rescue/overnight/run.py run  (state in {OUT})']
    (OUT / 'REPORT.md').write_text('\n'.join(o) + '\n'); log('REPORT written')


# ------------------------------------------------------------------ smoke (<=4 declared games)
def max_weight_diff(a, b):
    import torch
    sa = torch.load(a, map_location='cpu', weights_only=False)['state_dict']; sb = torch.load(b, map_location='cpu', weights_only=False)['state_dict']
    return max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)


def smoke():
    check_frozen(); starts = load_starts(); L = Ledger(); sm = starts['smoke']; ok = {}
    res, _ = dispatch(L, 'smoke', [('smoke|pv|0', pv_cmd(CHAMP), sm[0]['fight_id'], sm[0]['start'], {}), ('smoke|teacher|1', teacher_cmd(), sm[1]['fight_id'], sm[1]['start'], {})], keep_lines=True)
    ok['games'] = {k: (r['status'], r.get('won'), round(r['seconds'], 1)) for k, r in res.items()}
    if any(r['status'] != 'completed' for r in res.values()): raise Halt(f'smoke games {ok}')
    d = OUT / 'smoke'; d.mkdir(exist_ok=True); ei.encode([L.res['smoke|pv|0']['line']], d / 'rows.parquet')
    old = ei.OLD_SHARDS[:2]; ids = set()
    for f in old + [d / 'rows.parquet']: ids |= set(pq.read_table(f, columns=['fight_id'])['fight_id'].to_pylist())
    (d / 'split.json').write_text(json.dumps({i: 'train' for i in ids})); outs = {}
    for tag, extra in (('default', []), ('seed0', ['--rng-seed', '0']), ('seed1', ['--rng-seed', '1'])):
        shutil.rmtree(d / tag, ignore_errors=True)
        c = [str(ei.PY), str(frozen('train_fixed.py')), '--old', *map(str, old), '--new', str(d / 'rows.parquet'), '--split', str(d / 'split.json'), '--init', str(CHAMP / 'model.pt'),
             '--out', str(d / tag), '--update', '1', '--test-steps', '3', *extra]
        p = subprocess.run(c, capture_output=True, text=True)
        if p.returncode: raise Halt(f'smoke trainer {tag}: {p.stderr[-500:]}')
        outs[tag] = json.loads((d / tag / 'train.json').read_text())
    ok['loss'] = {t: [x['loss'] for x in o['log']] for t, o in outs.items()}
    ok['rng'] = {t: o['rng_seed_sequences'] for t, o in outs.items()}
    ok['weight_diff_default_vs_seed0'] = max_weight_diff(d / 'default/model.pt', d / 'seed0/model.pt')
    ok['weight_diff_seed0_vs_seed1'] = max_weight_diff(d / 'seed0/model.pt', d / 'seed1/model.pt')
    if ok['rng']['default'] != ok['rng']['seed0'] or ok['loss']['default'] != ok['loss']['seed0'] or ok['weight_diff_default_vs_seed0'] != 0.0:
        raise Halt(f'default not identical to explicit --rng-seed 0: {ok}')
    if ok['weight_diff_seed0_vs_seed1'] == 0.0 or ok['rng']['seed1'] == ok['rng']['seed0']: raise Halt('seed 1 does not change training')
    wjson(OUT / 'smoke-passed.json', {'at': now(), **ok}); log('SMOKE PASSED', ok)


def refreeze(reason):
    """Publish a revised source-hash manifest (prior manifest+sources archived as attemptK). Seeds/champion/old shards must be unchanged."""
    mp = OUT / 'manifest.json'; old = json.loads(mp.read_text()); k = old.get('attempt', 1)
    if (OUT / f'manifest.attempt{k}.json').exists(): raise SystemExit('attempt archive exists')
    arch = OUT / f'frozen/attempt{k}'; arch.mkdir(parents=True)
    shutil.copy2(mp, OUT / f'manifest.attempt{k}.json')
    for n in ('run.py', 'OVERNIGHT.md', 'train_fixed.py'):
        shutil.copy2(frozen(n), arch / n)
    for src, n in ((HERE / 'run.py', 'run.py'), (HERE.parent / 'OVERNIGHT.md', 'OVERNIGHT.md'), (HERE / 'train_fixed.py', 'train_fixed.py'), (HERE / 'supervise.sh', 'supervise.sh'), (HERE / 'test_run.py', 'test_run.py')):
        if src.exists(): shutil.copy2(src, frozen(n))
    new = dict(old); new['frozen_sha256'] = {n: sha(frozen(n)) for n in ('pv_worker', 'train_fixed.py', 'run.py', 'ei.py')}
    new['extra_sha256'] = {n: sha(frozen(n)) for n in ('supervise.sh', 'test_run.py') if frozen(n).exists()}
    new.update(attempt=k + 1, protocol_sha256=sha(frozen('OVERNIGHT.md')), supersedes_manifest_sha256=sha(mp), refreeze_reason=reason, refrozen_at=now(), games_before=(OUT / 'ledger.jsonl').exists())
    for key in ('starts_sha256', 'champion_sha256', 'old_shards_sha256'): assert new[key] == old[key]
    wjson(mp, new); log('REFROZEN attempt', k + 1, new['frozen_sha256'])


def verify():
    """Read-only recovery check used by the supervisor before any crash-resume."""
    check_frozen(warn_protocol=False); L = Ledger(); json.loads((OUT / 'state.json').read_text()) if (OUT / 'state.json').exists() else None
    bad = [k for k, r in L.res.items() if r['status'] == 'error']
    if bad: raise Halt(f'errored games in ledger: {bad[:3]}')
    log('VERIFY ok: intents', len(L.intent), 'results', len(L.res), 'orphans', len(set(L.intent) - set(L.res)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['prep', 'smoke', 'run', 'report', 'verify', 'refreeze']); ap.add_argument('--reason', default=''); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); LOGS.mkdir(parents=True, exist_ok=True)
    if a.cmd == 'prep': prep(); return
    if a.cmd == 'report': report(); return
    if a.cmd == 'refreeze':
        if not a.reason: raise SystemExit('--reason required')
        refreeze(a.reason); return
    def on_term(*_):
        HALT.set(); kill_active(); raise Halt('SIGTERM')
    signal.signal(signal.SIGTERM, on_term); signal.signal(signal.SIGINT, on_term)
    code = 0
    try: {'smoke': smoke, 'run': run, 'verify': verify}[a.cmd]()
    except Halt as e:
        code = 2; log('HALT', e); (OUT / 'ALERT').write_text(f'{now()} HALT {e}\n')
    except BaseException as e:
        code = 3; log('CRASH', e); traceback.print_exc(); (OUT / 'ALERT').write_text(f'{now()} CRASH {type(e).__name__}: {e}\n')
    finally:
        HALT.set() if code else None; DONE.set(); kill_active()
    if code and a.cmd == 'run':
        try: report()
        except Exception as e: log('report failed', e)
    sys.exit(code)


if __name__ == '__main__': main()
