"""Phase-2 deliberate-fit diagnostic: can update15 fit the 2,997 teacher policy rows? (capacity/plumbing only)

Two arms, identical except the per-state policy weight:
  weighted   : loss = sum(conf * CE) / n_policy_states_in_batch   (the trainer's --flat-policy-weighting objective)
  unweighted : loss = sum(CE) / n_policy_states_in_batch
Weighting changes the total gradient scale as well as relative state weights; differences are not attributable to
relative weighting alone. Both arms: update15 weights, FRESH AdamW (lr 3e-4, wd 0.01, not the saved moments: this is
not a replication), grad-norm clip 1 (the recipe value; fraction of clipped steps logged), batch 64, CPU, torch seed 0,
the same fixed per-epoch row permutations (np default_rng(0), drawn once and shared). Full passes over all policy rows.
No value loss, no checkpoint selection, no target sharpening, no argmax substitution. Policy-only gradients still
move the shared trunk, so value outputs can drift; reported descriptively.

Checkpoints saved after epochs 0 (initial), 1, 3, 10, 30. Evaluated on the training rows (in-sample) and on the
monitor validation rows (MCTS teacher policies on consumed monitor seeds: NOT a pristine held-out set).
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).parent))
import correction_absorption as C  # noqa: E402
from agents.combat.pv.model import NAMES, PolicyValue  # noqa: E402

CHECKPOINTS = (0, 1, 3, 10, 30)


def policy_rows(rows):
    """Indices (into rows.shard.rows / rows.meta) of states with a policy target."""
    return np.array([k for k, r in enumerate(rows.meta) if r['has_policy']])


def batch_conf(target, legal):
    n = legal.sum(-1).to(target.dtype)
    return ((n * target.max(-1).values - 1) / (n - 1).clamp(min=1)).clamp(0, 1)


def loss_fn(net, b, weighted):
    inputs, target, has = b[0], b[2], b[3]
    _, logits = net(*(inputs[n] for n in NAMES))
    w = has.to(target.dtype)
    if weighted: w = w * batch_conf(target, inputs['actions'][..., 260] != 0)
    ce = -(target * logits.log_softmax(-1)).sum(-1)
    return (ce * w).sum() / max(int(has.sum()), 1)


def input_key(shard, r):
    h = hashlib.sha256()
    for n in NAMES:
        v, o = shard.flat[n]; h.update(v[o[r]:o[r + 1]].tobytes())
    return h.hexdigest()


def joint_floor(rows, groups, weighted):
    """Exact minimum of mean_i w_i*KL(p_i || q) over the objective's achievable predictions. Rows with byte-identical
    inputs share q; within a row, moves with byte-identical action tokens share a logit (the action head scores each
    move token independently from shared trunk features; see test). KL is convex in q and decomposes, so per group
    the optimum is q* = class-average of the w-weighted mean target. Returns mean_i w_i KL_i over all rows (w=1 or
    conf)."""
    total = 0.0; n = 0
    for g in groups.values():
        ps = [np.asarray(rows.meta[k]['policy_target'], np.float64) for k in g]
        ws = np.array([C.confidence(p, len(p)) if weighted else 1.0 for p in ps]); n += len(g)
        if ws.sum() == 0: continue
        q = (ws[:, None] * np.stack(ps)).sum(0) / ws.sum()
        for c in C.token_classes(rows.meta[g[0]]['tokens'][:len(q)]): q[c] = q[c].mean()
        for w, p in zip(ws, ps):
            m = p > 0; total += w * float((p[m] * np.log(p[m] / q[m])).sum())
    return total / n


def irreducible(rows, idx):
    """Rows with byte-identical model inputs must share one prediction. Report conflicts in their argmax sets and the
    best achievable mean argmax-set agreement and the KL floor (each row's KL to the group's mean target)."""
    groups = collections.defaultdict(list)
    for k in idx: groups[input_key(rows.shard, rows.shard.rows[k])].append(k)
    best_hits = 0; kl_floor = 0.0; conflict_groups = 0; dup_rows = 0
    for g in groups.values():
        ps = [np.asarray(rows.meta[k]['policy_target'], np.float64) for k in g]
        tops = [set(np.flatnonzero(p == p.max())) for p in ps]
        best_hits += max(sum(a in t for t in tops) for a in range(len(ps[0])))
        mean = np.mean(ps, 0)
        kl_floor += sum(float((p[p > 0] * np.log(p[p > 0] / mean[p > 0])).sum()) for p in ps)
        if len(g) > 1:
            dup_rows += len(g)
            conflict_groups += not set.intersection(*tops)
    # Within one row, legal moves with byte-identical action tokens get identical logits; argmax-set membership is then
    # decided by index tiebreak if the target separates them.
    split_ties = 0; exact_hit_max = 0; exact_prob_max = 0.0; within_kl = 0.0; class_conflict = 0
    for k in idx:
        r = rows.meta[k]; p = np.asarray(r['policy_target'], np.float64); top = set(np.flatnonzero(p == p.max()))
        cls = C.token_classes(r['tokens'][:len(p)])
        split_ties += any(bool(set(c) & top) and not set(c) <= top for c in cls)
        # equal logits within a class: exact argmax (first index on ties) can hit only via a class's first member
        exact_hit_max += any(c[0] in top for c in cls)
        exact_prob_max += max(len(set(c) & top) / len(c) for c in cls)
        avg = p.copy()
        for c in cls: avg[c] = p[c].mean()
        within_kl += float((p[p > 0] * np.log(p[p > 0] / avg[p > 0])).sum())
    return dict(rows_where_identical_action_tokens_split_argmax_set=split_ties,
                exact_top_hit_max_under_equal_class_logits=exact_hit_max / len(idx),
                exact_top_prob_max_under_equal_class_logits=exact_prob_max / len(idx),
                within_row_kl_floor_mean=within_kl / len(idx),
                joint_kl_floor_unweighted=joint_floor(rows, groups, False),
                joint_kl_floor_conf_weighted=joint_floor(rows, groups, True),
                note='within_row and cross-row floors are separate diagnostics; joint_* is the exact joint minimum.',
                **dict(policy_rows=len(idx), distinct_inputs=len(groups), rows_in_duplicate_groups=dup_rows,
                groups_with_disjoint_argmax_sets=conflict_groups, max_argmax_agreement=best_hits / len(idx),
                kl_floor_mean=kl_floor / len(idx)))


def evaluate(net, rows, idx, ref_value, rescued=None):
    """Per-stratum means of policy metrics on policy rows; value drift vs reference outputs on the same rows."""
    net.eval(); out = collections.defaultdict(lambda: collections.defaultdict(list))
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            k = idx[i:i + 256]
            b = rows.shard.batch(rows.shard.rows[k])
            v, l = net(*(b[0][n] for n in NAMES))
            for j, kk in enumerate(k):
                r = rows.meta[kk]
                m = C.state_metrics(r, float(v[j]), l[j].numpy(), r['moves'][int(np.argmax(r['policy_target']))])
                m['value_drift'] = abs(float(v[j]) - ref_value[kk]); m['wkl'] = m['conf'] * m['kl']
                groups = ['all', r['stratum']]
                if rescued is not None: groups.append('rescued' if rescued[r['fight_id']] else 'unrescued')
                for g in groups:
                    for key in ['kl', 'wkl', 'ce', 'wce', 'top_hit', 'top_prob', 'top_margin', 'class_top_hit', 'class_top_prob',
                                'logit_tie_at_max', 'value_sq', 'value_drift']:
                        out[g][key].append(float(m[key]))
    return {g: dict({k: float(np.nanmean(v)) for k, v in d.items()}, states=len(d['kl'])) for g, d in out.items()}


def reference_values(net, rows, idx):
    ref = {}
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            k = idx[i:i + 256]
            v, _ = net(*(rows.shard.batch(rows.shard.rows[k])[0][n] for n in NAMES))
            ref.update(zip(k.tolist(), v.numpy().tolist()))
    return ref


def run_arm(arm, rows, idx, perms, val, vidx, rescued, out, deadline):
    torch.manual_seed(0)
    net = C.load_net(C.MODELS['update15'])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
    ref = reference_values(net, rows, idx); vref = reference_values(net, val, vidx)
    log = (out / arm / 'train.jsonl'); log.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with log.open('w') as f:
        for epoch in range(max(CHECKPOINTS) + 1):
            if epoch in CHECKPOINTS:
                torch.save(dict(state_dict=net.state_dict(), optimizer_state_dict=opt.state_dict(), epoch=epoch,
                                width=net.width, value_activation=net.value_activation),
                           out / arm / f'epoch{epoch:03d}.pt')
                rec = dict(arm=arm, epoch=epoch, train=evaluate(net, rows, idx, ref, rescued),
                           val=evaluate(net, val, vidx, vref))
                results.append(rec); f.write(json.dumps(rec) + '\n'); f.flush()
                print(json.dumps(dict(arm=arm, epoch=epoch, train_kl=rec['train']['all']['kl'],
                                      train_top_hit=rec['train']['all']['top_hit'], val_kl=rec['val']['all']['kl'],
                                      elapsed=round(time.time() - START, 1))), flush=True)
            if epoch == max(CHECKPOINTS): break
            net.train(); tot = 0.0; clipped = 0; steps = 0
            for i in range(0, len(idx), 64):
                b = rows.shard.batch(rows.shard.rows[idx[perms[epoch][i:i + 64]]])
                loss = loss_fn(net, b, arm == 'weighted')
                opt.zero_grad(); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); tot += loss.item(); clipped += bool(norm > 1.0); steps += 1
            f.write(json.dumps(dict(arm=arm, epoch=epoch + 1, train_loss=tot / steps, steps=steps,
                                    clipped_fraction=clipped / steps, elapsed=time.time() - START)) + '\n'); f.flush()
            if time.time() > deadline: raise TimeoutError(f'{arm}: wall limit hit after epoch {epoch + 1}')
    return results


START = time.time()


def references(rows, idx):
    """Per-stratum reference points for argmax-set mass/top-1 (unweighted objective):
    exact_target_mass_on_top: mass the teacher target itself puts on its argmax set (an exact-distribution reference,
      unattainable where identical action tokens split visits); qstar_*: the within-row class-averaged optimum q*
      (top-1 with first-index tiebreak); max_mass_on_top_under_class_logits: the largest argmax-set mass ANY
      prediction can reach under equal logits within identical-token classes (exceeds the exact-fit reference)."""
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for k in idx:
        r = rows.meta[k]; p = np.asarray(r['policy_target'], float); top = np.flatnonzero(p == p.max())
        cls = C.token_classes(r['tokens'][:len(p)]); q = p.copy()
        for c in cls: q[c] = q[c].mean()
        for g in ['all', r['stratum']]:
            d[g]['exact_target_mass_on_top'].append(p[top].sum()); d[g]['qstar_mass_on_top'].append(q[top].sum())
            d[g]['qstar_top1_first_index'].append(float(int(np.argmax(q)) in top))
            d[g]['max_top1_under_class_logits'].append(float(any(c[0] in top for c in cls)))
            d[g]['max_mass_on_top_under_class_logits'].append(max(len(set(c) & set(top)) / len(c) for c in cls))
    return {g: dict({k: float(np.mean(v)) for k, v in m.items()}, states=len(m['qstar_mass_on_top'])) for g, m in d.items()}


def reevaluate(out):
    """Re-score saved checkpoints (no training) into results-eval.json; used to add metrics after the run."""
    rows = C.Rows(C.TEACHER); idx = policy_rows(rows); val = C.Rows(C.VAL); vidx = policy_rows(val)
    fights = {r['fight_id']: r['won'] for r in pq.read_table(C.TC / 'teacher-play/fights-0.parquet', columns=['fight_id', 'won']).to_pylist()}
    init = C.load_net(C.MODELS['update15']); ref = reference_values(init, rows, idx); vref = reference_values(init, val, vidx)
    res = []
    for arm in ['weighted', 'unweighted']:
        for e in CHECKPOINTS:
            ck = torch.load(out / arm / f'epoch{e:03d}.pt', map_location='cpu', weights_only=False)
            net = PolicyValue(ck['width'], ck['value_activation']); net.load_state_dict(ck['state_dict'])
            res.append(dict(arm=arm, epoch=e, train=evaluate(net, rows, idx, ref, fights), val=evaluate(net, val, vidx, vref)))
    (out / 'results-eval.json').write_text(json.dumps(res, indent=1))
    (out / 'references.json').write_text(json.dumps(dict(train=references(rows, idx), val=references(val, vidx)), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--wall-seconds', type=float, default=600)
    ap.add_argument('--reevaluate', action='store_true', help='only re-score saved checkpoints')
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.reevaluate: torch.set_num_threads(4); reevaluate(a.out); return
    torch.set_num_threads(4); deadline = START + a.wall_seconds
    rows = C.Rows(C.TEACHER); idx = policy_rows(rows)
    val = C.Rows(C.VAL); vidx = policy_rows(val)
    fights = {r['fight_id']: r['won'] for r in pq.read_table(C.TC / 'teacher-play/fights-0.parquet', columns=['fight_id', 'won']).to_pylist()}
    rng = np.random.default_rng(0); perms = [rng.permutation(len(idx)) for _ in range(max(CHECKPOINTS))]
    meta = dict(init=str(C.MODELS['update15']), init_sha=C.sha(C.MODELS['update15']), teacher_rows=C.sha(C.TEACHER),
                val_rows=C.sha(C.VAL), script=C.sha(__file__), policy_rows=len(idx), val_policy_rows=len(vidx),
                perm_sha=hashlib.sha256(np.stack(perms).tobytes()).hexdigest(), checkpoints=CHECKPOINTS,
                optimizer='fresh AdamW lr3e-4 wd0.01', grad_clip=1.0, batch=64, threads=4, device='cpu',
                irreducible_train=irreducible(rows, idx))
    (a.out / 'config.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta['irreducible_train']), flush=True)
    res = []
    for arm in ['weighted', 'unweighted']:
        res += run_arm(arm, rows, idx, perms, val, vidx, fights, a.out, deadline)
    (a.out / 'results.json').write_text(json.dumps(res, indent=1))
    print('done', round(time.time() - START, 1), 's')


if __name__ == '__main__':
    main()
