"""Collect held-out combat positions with 2-draw full search and evaluate ThinkingHead checkpoints.

Example (collect 1,500 validation states + evaluate iter003 & iter004):
    .venv/bin/python -m apps.thinking_eval.eval run \\
        --collect-model /path/to/iter004 \\
        --eval-model iter003=/path/to/iter003/model.pt \\
        --eval-model iter004=/path/to/iter004/model.pt \\
        --fights /path/to/validation_champ.jsonl /path/to/validation_collector.jsonl /path/to/validation_automaton.jsonl \\
        --per-file 500 \\
        --out runs/schema=thinking_eval_v1/date=2026-11-27/id=val1500

Re-analyze an existing directory with additional checkpoints on CPU:
    .venv/bin/python -m apps.thinking_eval.eval analyze \\
        --dir runs/schema=thinking_eval_v1/date=2026-11-27/id=val1500 \\
        --eval-model iter003=/path/to/iter003/model.pt \\
        --device cpu
"""
import argparse
import json
import os
import random
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from agents.combat.nn import encoding_bridge
from agents.combat.nn.layout import LAYOUT
from agents.combat.nn.model import ModelSpec, THINKING_ROUNDS, build_model
from agents.combat.nn.tokens import MASKED_LOGIT
from agents.combat.player.play import gpu_environment

BUDGETS = tuple(LAYOUT.sample.thinking_budgets)
STATE_RECORD_HEADER = struct.Struct('<III')


def collector_executable():
    build = os.environ.get('STSRL_BUILD_DIR', 'main')
    return Path(__file__).resolve().parents[2] / 'build' / build / 'apps/thinking_eval/thinking_eval_collect'


def parse_named_path(spec):
    if '=' not in spec:
        path = Path(spec).resolve()
        return path.parent.name, path
    name, raw = spec.split('=', 1)
    return name.strip(), Path(raw).resolve()


def sample_fights(paths, per_file=None, seed=229):
    selected = []
    for idx, path in enumerate(paths):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        source = Path(path).stem
        if per_file is not None and per_file < len(rows):
            rows = random.Random(seed + idx).sample(rows, per_file)
        for row in rows:
            selected.append(dict(row, _source=source))
    return selected


def collect_positions(binary, collect_model, fights, out_dir, *, sims=2048, particles=64, considered_actions=128,
                      horizon_probe=True, probe_sims=160, probe_particles=16, grade_particles=16,
                      actors=9, slots=224, cpu=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = gpu_environment()
    if 'XDG_CACHE_HOME' not in env:
        cache = out_dir / '.cache'
        cache.mkdir(parents=True, exist_ok=True)
        env['XDG_CACHE_HOME'] = str(cache.resolve())

    states_bin = out_dir / 'states.bin'
    positions_path = out_dir / 'positions.jsonl'
    log_path = out_dir / 'collect.stderr.log'
    cmd = [
        str(Path(binary).resolve()),
        str(Path(collect_model).resolve()),
        str(states_bin.resolve()),
        '--sims', str(sims),
        '--particles', str(particles),
        '--considered-actions', str(considered_actions),
        '--horizon-probe' if horizon_probe else '--no-horizon-probe',
        '--probe-sims', str(probe_sims),
        '--probe-particles', str(probe_particles),
        '--grade-particles', str(grade_particles),
        '--actors', str(actors),
        '--slots', str(slots),
    ]
    if cpu:
        cmd.append('--cpu')

    sources = {i: f.get('_source', 'unknown') for i, f in enumerate(fights)}
    payload = ''.join(json.dumps({k: v for k, v in f.items() if k != '_source'}) + '\n' for f in fights)
    total = len(fights)
    t0 = time.monotonic()
    rows = []
    with log_path.open('w') as err_stream, subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err_stream, text=True, env=env
    ) as proc:
        proc.stdin.write(payload)
        proc.stdin.close()
        for done, line in enumerate(proc.stdout, 1):
            row = json.loads(line)
            row['source'] = sources.get(row['index'], 'unknown')
            rows.append(row)
            if done % max(1, min(100, total // 10 or 1)) == 0 or done == total:
                elapsed = time.monotonic() - t0
                rate = done / max(elapsed, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(f'[{elapsed:6.1f}s] collected {done}/{total} positions ({rate:.1f} pos/s, ETA {eta:.0f}s)',
                      flush=True)
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f'thinking_eval_collect exited {rc}; see {log_path}')

    rows.sort(key=lambda r: r['index'])
    positions_path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows


def load_collected(out_dir):
    out_dir = Path(out_dir)
    rows = [json.loads(line) for line in (out_dir / 'positions.jsonl').read_text().splitlines() if line.strip()]
    kept = [r for r in rows if r.get('decisions', 0) > 0]
    states, orders = [], []
    with (out_dir / 'states.bin').open('rb') as f:
        for r in kept:
            span = r['state_range']
            f.seek(span['begin'])
            raw = f.read(span['end'] - span['begin'])
            version, state_bytes, num_actions = STATE_RECORD_HEADER.unpack_from(raw, 0)
            if version != encoding_bridge.state_codec_version():
                raise ValueError(f'state codec mismatch: {version} != {encoding_bridge.state_codec_version()}')
            at = STATE_RECORD_HEADER.size
            states.append(raw[at:at + state_bytes])
            orders.append(np.frombuffer(raw, dtype=np.uint32, count=num_actions, offset=at + state_bytes).tolist())
    return kept, states, orders


def draw_regret(draw, prior_bits):
    q_map = {e['bits']: e['q'] for e in draw['edges'] if e['visits'] > 0 and e['q'] is not None}
    if not q_map:
        return 0.0
    best_q = max(q_map.values())
    return float(best_q - q_map.get(prior_bits, min(q_map.values())))


def extract_targets(rows):
    n = len(rows)
    vos_a = np.zeros(n, dtype=np.float64)
    vos_b = np.zeros(n, dtype=np.float64)
    disagree_ref = np.zeros(n, dtype=bool)
    has_probe = all('horizon_probe' in r for r in rows)
    h1_a = np.full((n, THINKING_ROUNDS), np.nan, dtype=np.float64)
    h1_b = np.full((n, THINKING_ROUNDS), np.nan, dtype=np.float64)
    end_a = np.full((n, THINKING_ROUNDS), np.nan, dtype=np.float64)
    end_b = np.full((n, THINKING_ROUNDS), np.nan, dtype=np.float64)
    probe_evals = np.zeros((n, THINKING_ROUNDS), dtype=np.float64)
    probe_disagree = np.zeros((n, THINKING_ROUNDS), dtype=bool)

    for i, r in enumerate(rows):
        d_a, d_b = r['draws']
        prior_bits = max(d_a['edges'], key=lambda e: e['logit'])['bits']
        vos_a[i] = draw_regret(d_a, prior_bits)
        vos_b[i] = draw_regret(d_b, prior_bits)
        disagree_ref[i] = d_a['action'] != prior_bits
        if has_probe:
            for k, rd in enumerate(r['horizon_probe']['rounds']):
                h1_a[i, k] = rd['gain_h1_a']
                h1_b[i, k] = rd['gain_h1_b']
                end_a[i, k] = rd['gain_end_a']
                end_b[i, k] = rd['gain_end_b']
                probe_evals[i, k] = rd['evals']
                probe_disagree[i, k] = rd['selected'] != rd['prior']

    return dict(
        vos_a=vos_a, vos_b=vos_b, disagree_ref=disagree_ref,
        has_probe=has_probe, h1_a=h1_a, h1_b=h1_b, end_a=end_a, end_b=end_b,
        probe_evals=probe_evals, probe_disagree=probe_disagree,
    )


@torch.inference_mode()
def evaluate_checkpoint(ckpt_path, states, orders, *, device='cpu', batch_size=128):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    spec = ModelSpec(**{k: v for k, v in ckpt['spec'].items() if k != 'encoding_version'})
    net = build_model(spec).to(device).eval()
    net.load_state_dict(ckpt['net'])

    net_no_ctx = build_model(spec).to(device).eval()
    net_no_ctx.load_state_dict(ckpt['net'])
    net_no_ctx.thinking.net[0].weight[:, :spec.dim].zero_()

    encoded = encoding_bridge.encode(states, orders, net.encoding)
    tokens_list = [np.frombuffer(e, dtype=np.float32).reshape(-1, LAYOUT.width) for e in encoded]
    n = len(tokens_list)

    thinking = np.zeros((n, THINKING_ROUNDS), dtype=np.float64)
    thinking_no_ctx = np.zeros((n, THINKING_ROUNDS), dtype=np.float64)
    values = np.zeros(n, dtype=np.float64)
    max_pis = np.zeros(n, dtype=np.float64)
    entropies = np.zeros(n, dtype=np.float64)

    by_len = np.argsort([len(t) for t in tokens_list])
    for start in range(0, n, batch_size):
        idx = by_len[start:start + batch_size]
        max_len = max(len(tokens_list[i]) for i in idx)
        x = np.zeros((len(idx), max_len, LAYOUT.width), dtype=np.float32)
        for j, i in enumerate(idx):
            t = tokens_list[i]
            x[j, :len(t)] = t
        xt = torch.from_numpy(x).to(device)
        v, logits, th = net.predict(xt)
        _, _, th_nc = net_no_ctx.predict(xt)
        legal = logits > MASKED_LOGIT / 2
        probs = logits.float().softmax(dim=-1)
        log_probs = logits.float().log_softmax(dim=-1).masked_fill(~legal, 0.0)
        ent = -(probs * log_probs).sum(dim=-1)
        for j, i in enumerate(idx):
            thinking[i] = th[j].cpu().numpy()
            thinking_no_ctx[i] = th_nc[j].cpu().numpy()
            values[i] = float(v[j].cpu())
            max_pis[i] = float(probs[j].max().cpu())
            entropies[i] = float(ent[j].cpu())

    return dict(
        thinking=thinking,
        thinking_no_ctx=thinking_no_ctx,
        value=values,
        max_pi=max_pis,
        entropy=entropies,
        free_vos=values * (1.0 - values) * (1.0 - max_pis),
        free_sqrt=np.sqrt(np.clip(values * (1.0 - values), 0.0, 1.0)) * (1.0 - max_pis),
        free_ent=values * (1.0 - values) * entropies,
    )


def topk_capture(scores, target, fracs=(0.05, 0.10, 0.20, 0.30), resamples=1000, seed=229):
    """Share of total `target` captured in the top `frac` ranked by `scores`, with bootstrap 95% CIs."""
    valid = np.isfinite(scores) & np.isfinite(target)
    scores, target = scores[valid], target[valid]
    n = len(target)
    total = float(target.sum())
    order = np.argsort(-scores, kind='mergesort')
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(resamples, n)) if n >= 10 and resamples > 0 else None

    out = {}
    for frac in fracs:
        k = max(1, int(round(frac * n)))
        share = float(target[order[:k]].sum() / total) if abs(total) > 1e-12 else float('nan')
        ci = None
        if boot_idx is not None and abs(total) > 1e-12:
            vals = []
            for b in range(resamples):
                bi = boot_idx[b]
                bs, bt = scores[bi], target[bi]
                denom = bt.sum()
                if denom > 1e-9:
                    bo = np.argsort(-bs, kind='mergesort')
                    vals.append(bt[bo[:k]].sum() / denom)
            if vals:
                ci = [float(x) for x in np.quantile(vals, [0.025, 0.975])]
        out[f'top_{int(round(frac * 100))}pct'] = dict(share=share, ci95=ci)
    return out


def decile_table(pred, targets_dict, disagree_mask, bins=10):
    n = len(pred)
    order = np.argsort(pred, kind='mergesort')
    out = []
    for d in range(bins):
        lo, hi = d * n // bins, (d + 1) * n // bins
        if hi <= lo:
            continue
        idx = order[lo:hi]
        entry = dict(
            decile=d + 1,
            count=int(len(idx)),
            pred_mean=float(pred[idx].mean()),
            pred_min=float(pred[idx].min()),
            pred_max=float(pred[idx].max()),
            disagree_rate=float(disagree_mask[idx].mean()),
        )
        for name, arr in targets_dict.items():
            entry[name] = float(np.nanmean(arr[idx]))
        out.append(entry)
    return out


def safe_corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) < 1e-12 or np.std(b[mask]) < 1e-12:
        return float('nan')
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def offline_voi_frontier(thinking_preds, free_scores, probe_evals, target_gains, lambdas):
    """Compare budget-aware ThinkingHead (`argmax_{k in 0..4} (pred_k - lambda*c_k)`) vs thresholded free score."""
    n = len(free_scores)
    budgets = np.array(BUDGETS, dtype=np.float64)
    head_curve = []
    for lam in lambdas:
        net_util = thinking_preds - lam * budgets[None, :]
        best_k = np.argmax(net_util, axis=1)
        enter = net_util[np.arange(n), best_k] > 0.0
        evals_spent = np.where(enter, probe_evals[np.arange(n), best_k], 0.0)
        realized = np.where(enter, target_gains[np.arange(n), best_k], 0.0)
        head_curve.append(dict(
            lambda_price=float(lam),
            entry_rate=float(enter.mean()),
            mean_extra_evals=float(evals_spent.mean()),
            mean_gain=float(np.nanmean(realized)),
        ))

    # Match the exact entry fractions using the free heuristic spending full probe (k=3)
    free_order = np.argsort(-free_scores, kind='mergesort')
    free_curve = []
    for pt in head_curve:
        k_top = int(round(pt['entry_rate'] * n))
        enter = np.zeros(n, dtype=bool)
        if k_top > 0:
            enter[free_order[:k_top]] = True
        evals_spent = np.where(enter, probe_evals[:, -1], 0.0)
        realized = np.where(enter, target_gains[:, -1], 0.0)
        free_curve.append(dict(
            entry_rate=float(enter.mean()),
            mean_extra_evals=float(evals_spent.mean()),
            mean_gain=float(np.nanmean(realized)),
        ))
    return dict(thinking_voi=head_curve, free_full_probe=free_curve)


def build_summary_and_report(rows, targets, model_evals):
    n = len(rows)
    vos_a, vos_b = targets['vos_a'], targets['vos_b']
    oracle_order = np.argsort(-vos_b)
    top10_n = max(1, int(round(0.10 * n)))
    oracle_top10 = float(vos_b[oracle_order[:top10_n]].sum() / max(vos_b.sum(), 1e-12))
    # Cross-draw ceiling: rank by Draw A, score on Draw B
    draw_a_ceiling = topk_capture(vos_a, vos_b, resamples=400)

    summary = dict(
        positions=n,
        by_source={src: sum(1 for r in rows if r.get('source') == src) for src in sorted({r.get('source', 'all') for r in rows})},
        reference=dict(
            mean_vos_b=float(vos_b.mean()),
            disagree_rate=float(targets['disagree_ref'].mean()),
            split_half_corr=safe_corr(vos_a, vos_b),
            oracle_top10_share=oracle_top10,
            draw_a_ceiling=draw_a_ceiling,
        ),
        models={},
    )

    lines = [
        f'=== ThinkingHead Held-Out State Evaluation (N={n} independent K>=2 positions) ===',
        f"Sources: {summary['by_source']}",
        f"Reference 2-draw Gumbel: mean VoS_B={vos_b.mean():+.5f}, prior!=ref={targets['disagree_ref'].mean():.1%}, "
        f"Draw A/B r={summary['reference']['split_half_corr']:.3f}, oracle top-10% concentration={oracle_top10:.1%}, "
        f"Draw-A->Draw-B noise ceiling top-10%={draw_a_ceiling['top_10pct']['share']:.1%}",
    ]

    if targets['has_probe']:
        h1_r = safe_corr(targets['h1_a'][:, -1], targets['h1_b'][:, -1])
        end_r = safe_corr(targets['end_a'][:, -1], targets['end_b'][:, -1])
        h1_end_r = safe_corr(targets['h1_a'][:, -1], targets['end_b'][:, -1])
        summary['horizon_probe'] = dict(
            budgets=list(BUDGETS),
            disagree_rates=[float(x) for x in targets['probe_disagree'].mean(axis=0)],
            mean_h1_b=[float(np.nanmean(targets['h1_b'][:, k])) for k in range(THINKING_ROUNDS)],
            mean_end_b=[float(np.nanmean(targets['end_b'][:, k])) for k in range(THINKING_ROUNDS)],
            b160_h1_split_corr=h1_r,
            b160_end_split_corr=end_r,
            b160_h1_vs_end_corr=h1_end_r,
        )
        lines.append(
            f"Horizon(160) teacher @ b=160: disagree={targets['probe_disagree'][:, -1].mean():.1%}, "
            f"mean Δ_B(H=1)={np.nanmean(targets['h1_b'][:, -1]):+.5f} (split r={h1_r:.3f}), "
            f"mean Δ_B(H=end)={np.nanmean(targets['end_b'][:, -1]):+.5f} (split r={end_r:.3f}, H1-vs-end r={h1_end_r:.3f})"
        )

    for model_name, ev in model_evals.items():
        th = ev['thinking']
        th_nc = ev['thinking_no_ctx']
        signals = [
            (f'{model_name}: thinking[3] (b=160)', th[:, -1]),
            (f'{model_name}: max_k thinking[k]', th.max(axis=1)),
            (f'{model_name}: no_ctx (6-summary only)', th_nc[:, -1]),
            ('free: v(1-v)(1-max_pi)', ev['free_vos']),
            ('free: sqrt(v(1-v))(1-max_pi)', ev['free_sqrt']),
            ('free: v(1-v)*entropy', ev['free_ent']),
        ]

        model_info = dict(
            pred_means=[float(x) for x in th.mean(axis=0)],
            capture_vos_b={},
            deciles_b160={},
        )
        lines.append(f'\n--- Model: {model_name} (pred means across {list(BUDGETS)}: '
                     + ', '.join(f'{x:+.5f}' for x in th.mean(axis=0)) + ') ---')

        # 1) Regret capture against deep Gumbel Draw B
        lines.append('  [Target A] Deep Gumbel Reference Regret Captured (Draw B VoS_B):')
        lines.append(f'    {"Signal":36s} {"r":>6s} {"Top 5%":>9s} {"Top 10% (95% CI)":>22s} {"Top 20%":>9s} {"Top 30%":>9s}')
        for sig_name, sig in signals:
            cap = topk_capture(sig, vos_b, resamples=500)
            r_val = safe_corr(sig, vos_b)
            model_info['capture_vos_b'][sig_name] = dict(corr=r_val, **cap)
            ci = cap['top_10pct']['ci95']
            ci_str = f"[{ci[0]:4.1%},{ci[1]:4.1%}]" if ci else ""
            lines.append(
                f"    {sig_name:36s} {r_val:+6.3f} {cap['top_5pct']['share']:8.1%} "
                f"{cap['top_10pct']['share']:7.1%} {ci_str:>13s} {cap['top_20pct']['share']:8.1%} {cap['top_30pct']['share']:8.1%}"
            )

        # 2) If Horizon probe present, also show H=1 and H=end capture
        t_dict = dict(vos_b=vos_b)
        if targets['has_probe']:
            t_dict['h1_b'] = targets['h1_b'][:, -1]
            t_dict['end_b'] = targets['end_b'][:, -1]
            model_info['capture_h1_b'] = {}
            model_info['capture_end_b'] = {}
            for label_key, target_arr, title in [
                ('capture_h1_b', targets['h1_b'][:, -1], 'Horizon(160) Out-of-Sample Gain Δ_B(H=1, Turn+1)'),
                ('capture_end_b', targets['end_b'][:, -1], 'Horizon(160) Unbiased Fight Outcome Gain Δ_B(H=end)'),
            ]:
                lines.append(f'  [Target] {title}:')
                for sig_name, sig in signals[:4]:
                    cap = topk_capture(sig, target_arr, resamples=500)
                    r_val = safe_corr(sig, target_arr)
                    model_info[label_key][sig_name] = dict(corr=r_val, **cap)
                    lines.append(
                        f"    {sig_name:36s} {r_val:+6.3f} {cap['top_5pct']['share']:8.1%} "
                        f"{cap['top_10pct']['share']:7.1%} {'':>13s} {cap['top_20pct']['share']:8.1%} {cap['top_30pct']['share']:8.1%}"
                    )

            lambdas = [0.0008, 0.0005, 0.0003, 0.0002, 0.00012, 0.00006]
            frontier = offline_voi_frontier(th, ev['free_vos'], targets['probe_evals'], targets['end_b'], lambdas)
            model_info['voi_frontier_end_b'] = frontier

        # 3) Decile calibration table
        deciles = decile_table(th[:, -1], t_dict, targets['disagree_ref'])
        model_info['deciles_b160'] = deciles
        lines.append('  Decile Calibration (sorted by thinking[3] @ b=160):')
        extra_hdr = '   Δ_B(H=1)  Δ_B(end)' if targets['has_probe'] else ''
        lines.append(f'    {"Dec":>3s} {"N":>5s} {"Pred Mean":>10s} {"Pred Range":>19s} {"VoS_B":>9s}{extra_hdr} {"Ref!=Prior":>10s}')
        for d in deciles:
            extra = f" {d['h1_b']:+9.5f} {d['end_b']:+9.5f}" if targets['has_probe'] else ''
            lines.append(
                f"    D{d['decile']:2d} {d['count']:5d} {d['pred_mean']:+10.5f} "
                f"[{d['pred_min']:+8.4f},{d['pred_max']:+8.4f}] {d['vos_b']:+9.5f}{extra} {d['disagree_rate']:9.1%}"
            )

        summary['models'][model_name] = model_info

    return summary, '\n'.join(lines) + '\n'


def cmd_run(args):
    fights = sample_fights(args.fights, per_file=args.per_file, seed=args.seed)
    print(f'Selected {len(fights)} fights across {len(args.fights)} file(s)', flush=True)
    collect_positions(
        args.binary,
        args.collect_model,
        fights,
        args.out,
        sims=args.sims,
        particles=args.particles,
        considered_actions=args.considered_actions,
        horizon_probe=args.horizon_probe,
        probe_sims=args.probe_sims,
        probe_particles=args.probe_particles,
        grade_particles=args.grade_particles,
        actors=args.actors,
        slots=args.slots,
        cpu=args.cpu,
    )
    args.dir = args.out
    cmd_analyze(args)


def cmd_analyze(args):
    rows, states, orders = load_collected(args.dir)
    targets = extract_targets(rows)
    model_evals = {}
    for spec in args.eval_model:
        name, ckpt_path = parse_named_path(spec)
        model_evals[name] = evaluate_checkpoint(ckpt_path, states, orders, device=args.device)

    summary, report = build_summary_and_report(rows, targets, model_evals)
    (Path(args.dir) / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (Path(args.dir) / 'report.txt').write_text(report)
    print(report, end='')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)

    p_run = sub.add_parser('run', help='Collect validation states with full search and evaluate ThinkingHead models')
    p_run.add_argument('--collect-model', type=Path, required=True, help='Exported model dir for trajectory + search')
    p_run.add_argument('--eval-model', action='append', required=True, help='NAME=PATH/model.pt (repeatable)')
    p_run.add_argument('--fights', type=Path, nargs='+', required=True, help='Fight JSONL files')
    p_run.add_argument('--per-file', type=int, default=500, help='Sampled fights per file (0 = all)')
    p_run.add_argument('--seed', type=int, default=229)
    p_run.add_argument('--out', type=Path, required=True)
    p_run.add_argument('--binary', type=Path, default=collector_executable())
    p_run.add_argument('--sims', type=int, default=2048)
    p_run.add_argument('--particles', type=int, default=64)
    p_run.add_argument('--considered-actions', type=int, default=128)
    p_run.add_argument('--horizon-probe', action=argparse.BooleanOptionalAction, default=True)
    p_run.add_argument('--probe-sims', type=int, default=160)
    p_run.add_argument('--probe-particles', type=int, default=16)
    p_run.add_argument('--grade-particles', type=int, default=16)
    p_run.add_argument('--actors', type=int, default=9)
    p_run.add_argument('--slots', type=int, default=224)
    p_run.add_argument('--cpu', action='store_true')
    p_run.add_argument('--device', default='cpu', help='PyTorch device for ThinkingHead readout (cpu or cuda)')

    p_an = sub.add_parser('analyze', help='Re-analyze an already collected directory with checkpoints')
    p_an.add_argument('--dir', type=Path, required=True)
    p_an.add_argument('--eval-model', action='append', required=True, help='NAME=PATH/model.pt (repeatable)')
    p_an.add_argument('--device', default='cpu')

    args = parser.parse_args()
    if getattr(args, 'per_file', None) == 0:
        args.per_file = None
    if args.command == 'run':
        cmd_run(args)
    else:
        cmd_analyze(args)


if __name__ == '__main__':
    main()
