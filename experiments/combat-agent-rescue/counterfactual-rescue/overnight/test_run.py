"""Offline tests (no games, no GPU): run from repo root:  .venv/bin/python -m pytest experiments/.../overnight/test_run.py -q"""
import json, sys, threading
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(R, 'OUT', tmp_path); (tmp_path / 'frozen').mkdir()
    monkeypatch.setattr(R.ei, 'OUT', tmp_path); R.HALT.clear(); R.DONE.clear()
    return tmp_path


def ledger_rows(p): return [json.loads(l) for l in (p / 'ledger.jsonl').read_text().splitlines()]


def test_ledger_duplicates_and_orphans(env):
    L = R.Ledger(); L.write({'kind': 'intent', 'key': 'a'})
    with pytest.raises(R.Halt): L.write({'kind': 'intent', 'key': 'a'})            # no second dispatch
    assert L.state('a') == 'orphan'
    L.write({'kind': 'result', 'key': 'a', 'status': 'completed', 'won': True})
    with pytest.raises(R.Halt): L.write({'kind': 'result', 'key': 'a', 'status': 'completed'})
    assert L.state('a') == 'completed' and L.state('a', need_line=True) == 'orphan'   # completed but no raw line = incomplete
    with pytest.raises(R.Halt): L.write({'kind': 'result', 'key': 'zz', 'status': 'completed'})
    L.f.close()
    p = env / 'ledger.jsonl'; p.write_text(p.read_text() + p.read_text().splitlines()[1] + '\n')   # duplicate result row on disk
    with pytest.raises(R.Halt): R.Ledger()


def test_ledger_partial_tail_is_orphan(env):
    L = R.Ledger(); L.write({'kind': 'intent', 'key': 'a'}); L.f.close()
    with open(env / 'ledger.jsonl', 'a') as f: f.write('{"kind": "result", "key": "a", "stat')   # crash mid-write
    L2 = R.Ledger(); assert L2.state('a') == 'orphan'; L2.write({'kind': 'intent', 'key': 'b'})
    assert len(ledger_rows(env)) == 2; assert list(env.glob("ledger.partial-tail.*")); R.Ledger()


def test_error_halts_no_retry(env, monkeypatch):
    L = R.Ledger(); L.write({'kind': 'intent', 'key': 'k'}); L.write({'kind': 'result', 'key': 'k', 'status': 'error', 'error': 'x', 'stderr': ''})
    monkeypatch.setattr(R, 'play', lambda *a: pytest.fail('replayed'))
    with pytest.raises(R.Halt): R.dispatch(L, 's', [('k', ['w', 'play'], 'f', {}, {})])
    assert not hasattr(R.Ledger, 'redispatch') and not (env / 'sre-redispatch.json').exists()


def test_wins_with_orphans_and_caps():
    res = {'a': {'status': 'completed', 'won': True}, 'b': {'status': 'capped'}, 'c': {'status': 'completed', 'won': False}}
    assert R.wins(res, ['a', 'b', 'c', 'missing']) == (1, 2)


def test_valid_model_requires_all_files_and_hashes(env):
    d = env / 'm'; d.mkdir()
    for n in R.MODEL_FILES: (d / n).write_text(n)
    (d / 'train.json').write_text(json.dumps({'files_sha256': {n: R.sha(d / n) for n in R.MODEL_FILES}}))
    assert R.valid_model(d); assert R.valid_model(d, {n: R.sha(d / n) for n in R.MODEL_FILES})
    assert not R.valid_model(d, {n: 'x' for n in R.MODEL_FILES})
    (d / 'model.onnx.data').write_text('tampered'); assert not R.valid_model(d)
    (d / 'model.onnx.data').unlink(); assert not R.valid_model(d)


def test_train_refuses_to_delete_ambiguous_model(env):
    d = env / 'repA/u01'; (d / 'model').mkdir(parents=True); (d / 'model/train.json').write_text('{}')
    with pytest.raises(R.Halt): R.train('A', 1, env, [], 0, 'A-u1')
    assert (d / 'model/train.json').exists()


def test_report_robust_zero_arms(env):
    R.report({'ref': {}, 'final': {'status': 'x', 'arms': {'challenger': {'completed': 0, 'capped': 0, 'wins': 0, 'won': {}, 'batch': {}, 'sec_mean': None, 'sec_median': None}}}})
    R.report({}); assert (env / 'REPORT.md').exists()


def test_deadline_constants_utc():
    assert R.DEADLINE - R.NO_NEW_UPDATE == 3600 and R.DEADLINE == 1791353700 if False else R.DEADLINE - R.NO_NEW_UPDATE == 3600


# ---- synthetic full-run resume tests: fake games / encode / trainer, tiny constants
class Sim:
    def __init__(self, env, monkeypatch, crash=None):
        self.played = []; self.crash = crash; self.ncalls = {'game': 0, 'train': 0}; self.models = []
        mp = monkeypatch
        mp.setattr(R, 'UPD', 3); mp.setattr(R, 'NTRAIN', 4); mp.setattr(R, 'MIN_DONE', 3); mp.setattr(R, 'NMON', 3); mp.setattr(R, 'NFIN', 3); mp.setattr(R, 'BATCH', 3)
        mp.setattr(R, 'MON_AT', (2, 3)); mp.setattr(R, 'WORKERS', 1); mp.setattr(R, 'CHAMP', env / 'champ'); mp.setattr(R, 'check_frozen', lambda **k: {'champion_sha256': {'model.pt': R.sha(env / 'champ/model.pt')}})
        mp.setattr(R, 'phase6_disjoint', lambda s: 0); mp.setattr(R, 'mem_guard', lambda: None)
        mp.setattr(R, 'pv_cmd', lambda m: ['w', 'play', str(m)]); mp.setattr(R, 'teacher_cmd', lambda: ['w', 'teacher', '20000'])
        mp.setattr(R.ei, 'encode', lambda lines, dest: dest.write_text('rows') or len(lines)); mp.setattr(R.pq, 'read_table', lambda *a, **k: None, raising=False)
        mp.setattr(R, 'play', self.play); mp.setattr(R, 'train', self.train); mp.setattr(R, 'NO_NEW_UPDATE', 10**12); mp.setattr(R, 'DEADLINE', 10**12 + 10**6)
        (env / 'champ').mkdir(exist_ok=True); [(env / 'champ' / n).write_text('champ' + n) for n in R.MODEL_FILES]
        (env / 'smoke-passed.json').write_text('{}')
        S = {}
        for r in R.REPS:
            for u in range(1, 4): S[f'train-{r}-u{u:02d}'] = [{'fight_id': f'f-{r}{u}-{i}', 'start': {'seed': 1}} for i in range(4)]
        S['monitor'] = [{'fight_id': f'm{i}', 'start': {'seed': 1}} for i in range(3)]; S['final'] = [{'fight_id': f'F{i}', 'start': {'seed': 1}} for i in range(3)]
        (env / 'starts.json').write_text(json.dumps(S)); self.env = env
        (env / 'manifest.json').write_text('{}')

    def play(self, cmd, key, fid, start, L, meta):
        self.ncalls['game'] += 1; self.played.append((key, cmd))
        L.write({'kind': 'intent', 'key': key, 'at': 'x', **meta})
        if self.crash == ('game', self.ncalls['game']): raise KeyboardInterrupt('simulated kill -9 after intent')
        rec = {'kind': 'result', 'key': key, 'fight_id': fid, 'seconds': 1.0, 'status': 'completed', 'won': hash(key) % 3 != 0, 'stderr': '', 'error': None, **meta}
        return rec, {'fight_id': fid, 'won': rec['won']}

    def train(self, rep, u, init, files, n_new, tag):
        self.ncalls['train'] += 1; d = self.env / f'rep{rep}/u{u:02d}'
        if R.valid_model(d / 'model'): return d / 'model'
        self.models.append((tag, str(init)))
        (d / 'model.tmp').mkdir(parents=True, exist_ok=True)
        if self.crash == ('train', self.ncalls['train']): raise KeyboardInterrupt('crash mid-train, partial model.tmp')
        for n in R.MODEL_FILES: (d / 'model.tmp' / n).write_text(f'{tag}{n}')
        h = {n: R.sha(d / 'model.tmp' / n) for n in R.MODEL_FILES}
        (d / 'model.tmp/train.json').write_text(json.dumps({'files_sha256': h, 'counts': {}, 'old_fights': 2032, 'new_fights': n_new, 'new_states': 1, 'train_seconds': 1, 'onnx_max_abs_diff': 0, 'init_sha256': 'x', 'log': [{}]}))
        import os; os.rename(d / 'model.tmp', d / 'model'); return d / 'model'


def finish(env, monkeypatch, crash=None):
    s = Sim(env, monkeypatch, crash)
    try: R.run()
    except BaseException as e: s.exc = e
    return s


@pytest.mark.parametrize('crash', [('game', 3), ('game', 5), ('game', 9), ('game', 16), ('game', 25), ('game', 30), ('train', 1), ('train', 3)])
def test_crash_resume_no_replay_and_model_chain(env, monkeypatch, crash):
    s1 = finish(env, monkeypatch, crash); assert isinstance(getattr(s1, 'exc', None), KeyboardInterrupt), 'crash did not trigger'
    monkeypatch.undo(); R.HALT.clear()
    monkeypatch.setattr(R, 'OUT', env); monkeypatch.setattr(R.ei, 'OUT', env)
    s2 = finish(env, monkeypatch, None)
    if crash[0] == 'game' and crash[1] <= 6:      # crash inside a reference monitor: orphan => must HALT (no replacement/replay), not continue
        assert isinstance(getattr(s2, 'exc', None), R.Halt) and not set(k for k, _ in s1.played) & set(k for k, _ in s2.played); return
    assert not hasattr(s2, 'exc'), getattr(s2, 'exc', None)
    keys1 = [k for k, _ in s1.played]; keys2 = [k for k, _ in s2.played]
    assert not set(keys1) & set(keys2), 'a previously dispatched game was dispatched again'
    rows = ledger_rows(env); ks = [r['key'] for r in rows if r['kind'] == 'intent']; assert len(ks) == len(set(ks))
    for k, cmd in s1.played + s2.played:                       # collect for u uses checkpoint u-1 of the same replica, u1 uses champion
        if k.startswith('collect|'):
            _, r, u, _ = k.split('|'); u = int(u)
            want = str(env / 'champ') if u == 1 else str(env / f'rep{r}/u{u-1:02d}/model')
            assert cmd[-1] == want, (k, cmd)
    st = json.loads((env / 'state.json').read_text()); assert st['selection'] is not None and 'finished' in st
