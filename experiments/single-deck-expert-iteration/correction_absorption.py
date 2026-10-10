"""Phase-1 diagnostic: was the Demon Form teacher correction absorbed at matched states?

Read-only on existing artifacts; CPU inference only; no gameplay, no training, no final seeds.

1. Provenance: sha256 of models/shards/worker/trainer source (current source; run-time snapshot unavailable).
2. Checkpoint identity: re-evaluate each saved model on the logged validation set and match epoch lines.
3. Matched-state metrics of update15 / control / correction on the 100 teacher replacement trajectories
   (primary) and on the 100 replaced learner-loss trajectories (secondary, different physical states).
4. Exact sampler reconstruction (model independent): runs the real agents.combat.pv.data Dataset code with
   batch materialisation stubbed out, verified against logged per-epoch counts; per-state exposure/policy mass.
5. ONNX vs PyTorch parity on a subset (native C++ evaluator parity NOT checked).

python experiments/single-deck-expert-iteration/correction_absorption.py --out runs/.../id=demon-form-correction-absorption-v1/out
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
import torch

import agents.combat.pv.data as D
from agents.combat.pv.model import NAMES, PolicyValue

DATE = Path('runs/schema=combat_v4/date=2026-10-06')
SRC = DATE / 'id=single-deck-demon-form-v1/out'
TC = DATE / 'id=demon-form-teacher-correction-v1/out'
UP = DATE / 'id=demon-form-correction-uptake-v1/out'
MODELS = {'update15': TC / 'frozen/model.pt', 'control': TC / 'control/model/model.pt',
          'correction': TC / 'correction/model/model.pt'}
LOGS = {'update15': SRC / 'iter015/logs/train.log', 'control': TC / 'control/logs/train.log',
        'correction': TC / 'correction/logs/train.log'}
CARD = {11: 'Anger', 15: "Ascender's Bane", 25: 'Bash', 33: 'Bite', 65: 'Clash', 67: 'Cleave', 83: 'Corruption',
        104: 'Defend', 107: 'Demon Form', 124: 'Dual Wield', 179: 'Headbutt', 197: 'Intimidate',
        311: 'Spot Weakness', 329: 'Sword Boomerang'}  # sts_lightspeed include/constants/Cards.h
SETUP = {107, 83, 311, 124}  # Demon Form, Corruption, Spot Weakness, Dual Wield (powers / setup skill / copy)
KIND_CARD, KIND_SELECT, KIND_END = 0, 2, 4  # environments/combat/encoding.hpp EncodedActionKind
TASK_DUAL_WIELD, TASK_HEADBUTT = 4, 10       # sts_lightspeed CardSelectTask enum order
SHARDS = [SRC / f'iter{i:03d}/rows/rows.parquet' for i in range(6, 16)]
TEACHER = TC / 'teacher-rows/rows.parquet'
VAL = SRC / 'monitor-rows/rows.parquet'
BOOT_SEED = 20261006
ROWS = {}
NBOOT = 2000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_net(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    net = PolicyValue(ck['width'], ck.get('value_activation', 'softplus'))
    net.load_state_dict(ck['state_dict']); net.eval()
    return net


def confidence(p, n):
    """Same formula as agents.combat.pv.train.policy_confidence (scalar form)."""
    return float(np.clip((n * p.max() - 1) / max(n - 1, 1), 0, 1))


def stratum(tokens, kind_task):
    """Predefined strategic strata from recorded action semantics (not outcomes, not targets)."""
    kind, task = kind_task
    if kind == KIND_SELECT and task == TASK_DUAL_WIELD: return 'dual_wield_select'
    if kind == KIND_SELECT and task == TASK_HEADBUTT: return 'headbutt_select'
    if kind == KIND_SELECT: return 'other_select'
    cards = {int(t[8]) for t in tokens if t[0] == KIND_CARD}
    if cards & SETUP: return 'setup_legal'
    return 'other_play'


def label(t):
    if t[0] == KIND_END: return 'end turn'
    name = CARD.get(int(t[8]), str(int(t[8]))) + ('+' if t[12] else '')
    return name if t[0] == KIND_CARD else 'select ' + name


class Rows:
    """One rows.parquet restricted to fight IDs; per-row metadata plus Shard for padded batches."""

    def __init__(self, path, fids=None):
        meta = pq.read_table(path, columns=['fight_id', 'step', 'turn', 'won', 'moves', 'policy_target', 'has_policy',
                                            'n_actions']).to_pylist()
        ids = sorted({r['fight_id'] for r in meta}) if fids is None else fids
        self.shard = D.Shard(path, False, {r['fight_id']: 'train' for r in meta}, set(ids))
        self.meta = [meta[i] for i in self.shard.rows]
        values, offsets = self.shard.flat['actions']
        for i, r in zip(self.shard.rows, self.meta):
            r['tokens'] = values[offsets[i]:offsets[i + 1]].reshape(-1, 261)
            r['stratum'] = stratum(r['tokens'], (int(r['tokens'][0, 0]), int(r['tokens'][0, 1])))


def forward(net, rows, size=256):
    vals, logits = [], []
    with torch.no_grad():
        for i in range(0, len(rows.shard.rows), size):
            b = rows.shard.batch(rows.shard.rows[i:i + size])
            v, l = net(*(b[0][n] for n in NAMES))
            vals.append(v.numpy()); logits += [x for x in l.numpy()]
    return np.concatenate(vals), logits


def state_metrics(r, value, logit, played):
    n = len(r['moves']); z = logit[:n].astype(np.float64)
    logq = z - z.max() - np.log(np.exp(z - z.max()).sum()); q = np.exp(logq)
    out = dict(value=float(value), value_sq=float(((value - 100 * r['won']) / 100) ** 2))
    a = r['moves'].index(played)
    others = np.delete(z, a)
    out.update(played_prob=float(q[a]), played_rank=int(1 + (z > z[a]).sum()),
               played_margin=float(z[a] - others.max()) if len(others) else float('nan'))
    # Network-indistinguishable class: legal moves with byte-identical action tokens (not physical equivalence).
    same = [j for j in range(n) if np.array_equal(r['tokens'][j], r['tokens'][a])]
    out['played_class_prob'] = float(q[same].sum())
    if r['has_policy']:
        p = np.asarray(r['policy_target'], np.float64)
        ce = float(-(p * logq).sum()); h = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        best = np.flatnonzero(p == p.max()); rest = np.setdiff1d(np.arange(n), best)
        out.update(ce=ce, kl=ce - h, conf=confidence(p, n), wce=ce * confidence(p, n), top_prob=float(q[best].sum()),
                   top_rank=int(1 + (z[rest] > z[best].max()).sum()), top_hit=bool(z.argmax() in best),
                   top_margin=float(z[best].max() - z[rest].max()) if len(rest) else float('nan'),
                   n_top=len(best), played_is_top=bool(a in best))
        # Network-indistinguishable classes (byte-identical action tokens get identical logits). Secondary metrics:
        # class target = summed visits; not a claim of physical equivalence.
        cls = token_classes(r['tokens'][:n])
        pc = np.array([p[c].sum() for c in cls]); qc = np.array([q[c].sum() for c in cls])
        bc = np.flatnonzero(np.isclose(pc, pc.max(), rtol=0, atol=1e-7))
        out.update(class_top_prob=float(qc[bc].sum()), class_top_hit=bool(qc.argmax() in bc),
                   logit_tie_at_max=bool((np.abs(z - z.max()) < 1e-6).sum() > 1))
    return out


def token_classes(tokens):
    """Partition move indices by byte-identical action token rows, in first-index order."""
    seen = {}
    for j, t in enumerate(tokens): seen.setdefault(t.tobytes(), []).append(j)
    return list(seen.values())


def evaluate_set(name, rows, played_lookup, nets, won_by_fight):
    """Per-state records for each model: list of dicts with fight/stratum/metrics."""
    rec = []
    outputs = {m: forward(net, rows) for m, net in nets.items()}
    for i, r in enumerate(rows.meta):
        played = played_lookup[r['fight_id']][r['step']]
        base = dict(set=name, fight_id=r['fight_id'], step=r['step'], turn=r['turn'], stratum=r['stratum'],
                    has_policy=bool(r['has_policy']), won=bool(r['won']), rescued=won_by_fight.get(r['fight_id']),
                    played=label(r['tokens'][r['moves'].index(played)]), n_legal=len(r['moves']))
        if r['has_policy']:
            p = np.asarray(r['policy_target']); base['target_top'] = label(r['tokens'][int(p.argmax())])
        for m in nets:
            base[m] = state_metrics(r, outputs[m][0][i], outputs[m][1][i], played)
        rec.append(base)
    return rec


METRICS = ['value_sq', 'ce', 'kl', 'wce', 'top_prob', 'top_rank', 'top_margin', 'top_hit', 'played_prob',
           'played_rank', 'played_margin', 'played_class_prob', 'class_top_prob', 'class_top_hit', 'logit_tie_at_max']


def fight_table(rec, getter, fights):
    """Per-fight (sum, count) arrays of a per-state quantity; NaN/None skipped."""
    idx = {f: k for k, f in enumerate(fights)}; s = np.zeros(len(fights)); n = np.zeros(len(fights))
    for r in rec:
        v = getter(r)
        if v is None or (isinstance(v, float) and np.isnan(v)): continue
        s[idx[r['fight_id']]] += float(v); n[idx[r['fight_id']]] += 1
    return s, n


def agg(s, n, take=None):
    """(decision-level mean, fight-balanced mean) over the fight multiset `take` (indices)."""
    if take is not None: s, n = s[take], n[take]
    ok = n > 0
    if not ok.any(): return float('nan'), float('nan')
    return float(s.sum() / n.sum()), float((s[ok] / n[ok]).mean())


def summarize(rec, models, pairs):
    """Means per model and paired per-state deltas; fight-cluster bootstrap (fixed models, seed BOOT_SEED)."""
    fights = sorted({r['fight_id'] for r in rec})
    rng = np.random.default_rng(BOOT_SEED)
    boots = rng.integers(0, len(fights), (NBOOT, len(fights)))
    out = {}
    for metric in METRICS:
        row = {}
        for m in models:
            s, n = fight_table(rec, lambda r: r[m].get(metric), fights)
            d, f = agg(s, n); row[m] = dict(decision=d, fight_balanced=f, states=int(n.sum()), fights=int((n > 0).sum()))
        for a, b in pairs:
            def delta(r):
                x, y = r[a].get(metric), r[b].get(metric)
                return None if x is None or y is None else float(x) - float(y)
            s, n = fight_table(rec, delta, fights)
            d, f = agg(s, n); bs = np.array([agg(s, n, t) for t in boots])
            row[f'{a}-{b}'] = dict(decision=d, fight_balanced=f, states=int(n.sum()), fights=int((n > 0).sum()),
                                   decision_ci95=np.nanpercentile(bs[:, 0], [2.5, 97.5]).tolist(),
                                   fight_balanced_ci95=np.nanpercentile(bs[:, 1], [2.5, 97.5]).tolist())
        out[metric] = row
    return out


# ---------------- sampler reconstruction ----------------

def reconstruct(paths, split, train_ids, stream, mix, epochs=3):
    """Run the real Dataset sampling/shuffle code with batch tensors stubbed. Returns per-epoch lists of batches,
    each a list of (path, row) pairs. Valid only because the RNG call sequence is model-independent (checked by
    matching logged counts)."""
    orig_batch, orig_merge = D.Shard.batch, D.merge
    D.Shard.batch = lambda self, idx: [(self, np.asarray(idx))]
    D.merge = lambda parts: [x for p in parts for x in p]
    try:
        ds = D.Dataset(paths, False, stream, split, set(train_ids), 64, mix)
        next(ds.batches(2))  # train.py draws an export example first (rng=None; no global RNG use)
        rng = np.random.default_rng(0)
        epochs_out = []
        for _ in range(epochs):
            batches = []
            for b in ds.batches(64, rng):
                batches.append([(s, idx) for s, idx in b])
            epochs_out.append(batches)
        return epochs_out
    finally:
        D.Shard.batch, D.merge = orig_batch, orig_merge


def shard_conf(shard):
    """Per-row concentration weight × has_policy, exactly as train.evaluate computes it."""
    vals, offs = shard.policy; n = shard.counts['actions']
    conf = np.zeros(len(n))
    for r in range(len(n)):
        p = vals[offs[r]:offs[r + 1]]
        if shard.has_policy[r] and len(p): conf[r] = confidence(p, int(n[r]))
    return conf


def exposure(epochs_out, logs):
    """Per-(shard id, row) sampled count and policy gradient mass (conf / policy states in its batch), per epoch;
    plus check against logged train counts."""
    cache = {}
    per = collections.defaultdict(lambda: np.zeros((len(epochs_out), 2)))
    checks = []
    for e, batches in enumerate(epochs_out):
        states = pol = eff = 0
        for b in batches:
            rows = [(s, int(i)) for s, idx in b for i in idx]
            for s, _ in b:
                if id(s) not in cache: cache[id(s)] = (s, shard_conf(s))  # keep s alive: no id reuse
            npol = sum(bool(s.has_policy[i]) for s, i in rows)
            for s, i in rows:
                c = cache[id(s)][1][i]
                k = (s.fight_ids[i], i)
                per[k][e, 0] += 1; per[k][e, 1] += c / max(npol, 1)
                states += 1; pol += bool(s.has_policy[i]); eff += c
        lg = logs[e]['train']
        checks.append(dict(epoch=e, states=states, logged_states=lg['states'], policy_states=pol,
                           logged_policy_states=lg['policy_states'], effective=float(eff),
                           logged_effective=lg.get('effective_policy_states'),
                           match=bool(states == lg['states'] and pol == lg['policy_states']
                           and abs(eff - lg.get('effective_policy_states', eff)) < 0.5)))
    return per, checks


def read_log(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.startswith('{"epoch"')]


def val_score(net, path, split):
    rows = D.Dataset([path], True, False, split)
    v = p = n = npol = 0
    from agents.combat.pv.train import evaluate
    with torch.no_grad():
        for b in rows.batches(64):
            a, b2, _, s, sp = evaluate(net, b, 'cpu', flat_policy_weighting=True)
            v += a.item(); p += b2.item(); n += s; npol += sp
    return dict(value_loss=v / n, policy_loss=p / npol, states=n, policy_states=npol)


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--skip-sampler', action='store_true')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    res = {}
    # 1. provenance
    src = ['agents/combat/pv/train.py', 'agents/combat/pv/data.py', 'agents/combat/pv/model.py',
           'apps/run_rl/teacher_correction.py', 'apps/run_rl/single_deck.py', 'apps/run_rl/combat_loop.py',
           'experiments/single-deck-expert-iteration/correction_absorption.py']
    res['provenance'] = dict(
        models={k: sha(v) for k, v in MODELS.items()},
        update15_source_checkpoint=sha(SRC / 'iter015/model/model.pt'),
        correction_config=json.loads((TC / 'config.json').read_text()),
        worker=sha(TC / 'frozen/pv_worker'),
        shards={str(p): sha(p) for p in SHARDS + [TEACHER, VAL]},
        current_source={p: sha(p) for p in src},
        git_head=subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
        git_status_source=subprocess.run(['git', 'status', '--short', '--', *src], capture_output=True, text=True).stdout,
        note='No run-time source snapshot exists; current-source hashes do NOT prove the code that ran.')
    nets = {k: load_net(v) for k, v in MODELS.items()}
    split = json.loads((TC / 'split.json').read_text())
    # 2. checkpoint identity on validation set
    res['checkpoint'] = {}
    for m, net in nets.items():
        vs = val_score(net, VAL, split); lg = read_log(LOGS[m])
        dist = [abs(l['val']['value_loss'] - vs['value_loss']) + abs(l['val']['policy_loss'] - vs['policy_loss']) for l in lg]
        res['checkpoint'][m] = dict(recomputed=vs, logged=[dict(epoch=l['epoch'], value=l['val']['value_loss'],
                                    policy=l['val']['policy_loss'], score=l['val']['value_loss'] + l['val']['policy_loss'])
                                    for l in lg], matched_epoch=int(np.argmin(dist)), match_distance=float(min(dist)))
    # 3. matched-state evaluation
    sel = json.loads((TC / 'selection.json').read_text())
    teacher_fights = {r['fight_id']: r for r in pq.read_table(TC / 'teacher-play/fights-0.parquet').to_pylist()}
    rescued = {fid: teacher_fights[fid]['won'] for fid in teacher_fights}
    played = {fid: r['actions'] for fid, r in teacher_fights.items()}
    teach = Rows(TEACHER)
    assert len({r['fight_id'] for r in teach.meta}) == 100
    rec = evaluate_set('teacher', teach, played, nets, rescued)
    learner_played = {}
    for i in range(6, 16):
        for r in pq.read_table(SRC / f'iter{i:03d}/collect/fights-0.parquet', columns=['fight_id', 'actions']).to_pylist():
            if r['fight_id'] in sel['replacement']: learner_played[r['fight_id']] = r['actions']
    lrec = []
    by_shard = collections.defaultdict(list)
    for fid in sel['selected']: by_shard[int(fid.split(':')[1].split('-')[1])].append(fid)
    lres = {fid: rescued[sel['replacement'][fid]] for fid in sel['selected']}
    for it, fids in sorted(by_shard.items()):
        lrec += evaluate_set('learner_replaced', Rows(SRC / f'iter{it:03d}/rows/rows.parquet', fids), learner_played, nets, lres)
    models = list(MODELS); pairs = [('correction', 'control'), ('correction', 'update15'), ('control', 'update15')]
    res['matched'] = {}
    for setname, rr in [('teacher', rec), ('learner_replaced', lrec)]:
        groups = {'all': rr, 'rescued': [r for r in rr if r['rescued']], 'unrescued': [r for r in rr if not r['rescued']]}
        for s in ['dual_wield_select', 'setup_legal', 'headbutt_select', 'other_play']:
            groups['stratum:' + s] = [r for r in rr if r['stratum'] == s]
        groups['turn<=2'] = [r for r in rr if r['turn'] <= 2]
        res['matched'][setname] = {g: summarize(x, models, pairs) for g, x in groups.items() if x}
    with (a.out / 'states.jsonl').open('w') as f:
        for r in rec + lrec: f.write(json.dumps(r) + '\n')
    # teacher target vs played and DW choice tables (descriptive)
    dw = collections.Counter(); dw_played = collections.Counter()
    for r in rec:
        if r['stratum'] == 'dual_wield_select' and r['has_policy']:
            dw[(r['target_top'], r['played'])] += 1
    res['teacher_dw_target_vs_played'] = {f'{k[0]} | played {k[1]}': v for k, v in dw.most_common()}
    res['teacher_played_is_target_top'] = dict(
        states=sum(r['has_policy'] for r in rec), equal=sum(r['update15'].get('played_is_top', False) for r in rec))
    # 4. sampler reconstruction & exposure
    if not a.skip_sampler:
        ids = json.loads((SRC / 'iter015/train-fights.json').read_text())
        arms = {'control': (SHARDS + [VAL], json.loads((TC / 'control/train-fights.json').read_text()), False, True),
                'correction': (SHARDS + [TEACHER, VAL], json.loads((TC / 'correction/train-fights.json').read_text()), False, True),
                'update15': (SHARDS + [VAL], ids, True, False)}
        split15 = json.loads((SRC / 'split.json').read_text())
        res['exposure'] = {}
        teacher_ids = set(sel['replacement'].values()); replaced = set(sel['selected'])
        for arm, (paths, tids, stream, mix) in arms.items():
            ep = reconstruct(paths, split15 if arm == 'update15' else split, tids, stream, mix)
            per, checks = exposure(ep, read_log(LOGS[arm]))
            sel_epoch = res['checkpoint'][arm]['matched_epoch']
            strata = {}
            for p in set(paths):  # stratum per row of every shard
                rows = ROWS[p] if p in ROWS else ROWS.setdefault(p, Rows(p))
                for i, r in zip(rows.shard.rows, rows.meta): strata[(r['fight_id'], int(i))] = r['stratum']
            agg = collections.defaultdict(lambda: np.zeros(5))
            for (fid, i), v in per.items():
                src_kind = 'teacher_replacement' if fid in teacher_ids else 'replaced_learner' if fid in replaced else 'retained_learner'
                for key in [(src_kind, 'all'), (src_kind, strata[(fid, i)])]:
                    agg[key] += [v[:, 0].sum(), v[:, 1].sum(), v[:sel_epoch + 1, 0].sum(), v[:sel_epoch + 1, 1].sum(), 1]
            # unsampled rows
            total_rows = collections.Counter(); tset = set(tids)
            for (fid, i), s in strata.items():
                if fid in tset:
                    k = 'teacher_replacement' if fid in teacher_ids else 'replaced_learner' if fid in replaced else 'retained_learner'
                    total_rows[(k, 'all')] += 1; total_rows[(k, s)] += 1
            res['exposure'][arm] = dict(checks=checks, selected_epoch=sel_epoch, batches_per_epoch=[len(b) for b in ep],
                                        table={f'{k[0]}|{k[1]}': dict(rows=total_rows[k], sampled_rows=int(v[4]),
                                               samples_3ep=int(v[0]), policy_mass_3ep=float(v[1]),
                                               samples_to_selected=int(v[2]), policy_mass_to_selected=float(v[3]))
                                               for k, v in sorted(agg.items())})
            # DW selection: policy-gradient mass x target probability per selectable card (soft-target "pull"),
            # by data source; plus mean concentration weight per source over all rows (unsampled included).
            pull = collections.defaultdict(float); confs = collections.defaultdict(list)
            for p in set(paths):
                rows = ROWS[p]; conf = shard_conf(rows.shard)
                for i, r in zip(rows.shard.rows, rows.meta):
                    if r['fight_id'] not in tset: continue
                    src_kind = 'teacher_replacement' if r['fight_id'] in teacher_ids else 'replaced_learner' if r['fight_id'] in replaced else 'retained_learner'
                    if r['has_policy']: confs[(src_kind, r['stratum'])].append(conf[i]); confs[(src_kind, 'all')].append(conf[i])
                    if r['stratum'] != 'dual_wield_select' or not r['has_policy']: continue
                    v = per.get((r['fight_id'], int(i)))
                    if v is None: continue
                    for j, pj in enumerate(r['policy_target']):
                        pull[(src_kind, '3ep', label(r['tokens'][j]))] += v[:, 1].sum() * pj
                        pull[(src_kind, 'selected', label(r['tokens'][j]))] += v[:sel_epoch + 1, 1].sum() * pj
            res['exposure'][arm]['dw_target_pull'] = {'|'.join(k): v for k, v in sorted(pull.items())}
            res['exposure'][arm]['mean_conf_policy_rows'] = {'|'.join(k): dict(n=len(v), mean=float(np.mean(v)))
                                                              for k, v in sorted(confs.items())}
    # 5. ONNX parity on 256 teacher states
    import onnxruntime as ort
    res['onnx_parity'] = {}
    b = teach.shard.batch(teach.shard.rows[:256])
    for m, path in MODELS.items():
        onnx = {'update15': SRC / 'iter015/model/model.onnx', 'control': TC / 'control/model/model.onnx',
                'correction': TC / 'correction/model/model.onnx'}[m]
        s = ort.InferenceSession(str(onnx), providers=['CPUExecutionProvider'])
        v, l = s.run(None, {n: b[0][n].numpy() for n in NAMES})
        with torch.no_grad(): tv, tl = nets[m](*(b[0][n] for n in NAMES))
        mask = b[0]['actions'][..., 260].numpy() != 0
        res['onnx_parity'][m] = dict(max_value_diff=float(np.abs(v - tv.numpy()).max()),
                                     max_logit_diff=float(np.abs(l - tl.numpy())[mask].max()), states=256)
    (a.out / 'results.json').write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(dict(checkpoint={k: v['matched_epoch'] for k, v in res['checkpoint'].items()},
                          onnx=res['onnx_parity'],
                          sampler={k: [c['match'] for c in v['checks']] for k, v in res.get('exposure', {}).items()}), indent=1, default=str))


if __name__ == '__main__':
    main()
