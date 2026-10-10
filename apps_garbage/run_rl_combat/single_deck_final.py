"""One-shot reserved single-deck final test, fixed selected model vs MCTS. No training."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq

from apps.run_rl.combat_loop import play, sha
from apps.run_rl.single_deck import score, wilson


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True, help='completed single_deck managed run')
    p.add_argument('--model', type=Path, required=True, help='preselected specialist model directory')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=10)
    a = p.parse_args()
    if not 1 <= a.workers <= 10:
        p.error('1–10 workers required')
    parent = a.parent.resolve()
    source = parent / 'out'
    metadata = json.loads((parent / 'run.json').read_text())
    if metadata['status'] != 'done':
        raise ValueError('training experiment must be finished before final test')
    seeds = source / 'final-reserved.parquet'
    worker_source = source / 'frozen/pv_worker'
    rows = pq.read_table(seeds).to_pylist()
    final_seeds = {r['start']['seed'] for r in rows}
    if len(rows) != 600 or len(final_seeds) != 600:
        raise ValueError('expected 600 distinct reserved final starts')
    for file in [source / 'bootstrap.parquet', source / 'monitor.parquet', *source.glob('iter*/starts.parquet')]:
        if final_seeds & {r['start']['seed'] for r in pq.read_table(file).to_pylist()}:
            raise ValueError(f'final seed overlap with {file}')
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'controller.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = dict(parent=str(parent), selected_model=str(a.model.resolve()), workers=a.workers,
                  starts_sha=sha(seeds), worker_sha=sha(worker_source), model_sha=sha(a.model / 'model.pt'),
                  onnx_sha=sha(a.model / 'model.onnx'),
                  onnx_data_sha=sha(a.model / 'model.onnx.data') if (a.model / 'model.onnx.data').exists() else None,
                  n=600, learned_sims=2000, teacher_sims=20000, training=False)
    path = out / 'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('cannot change selected model/final-test settings on resume')
    path.write_text(json.dumps(config, indent=2))
    frozen = out / 'frozen'
    frozen.mkdir(exist_ok=True)
    worker = frozen / 'pv_worker'
    model = frozen / 'model'
    if not worker.exists(): shutil.copy2(worker_source, worker)
    if not model.exists(): shutil.copytree(a.model, model)
    starts = out / 'starts.parquet'
    if not starts.exists(): shutil.copy2(seeds, starts)
    if sha(worker) != config['worker_sha'] or sha(starts) != config['starts_sha']:
        raise ValueError('frozen worker/starts checksum mismatch')
    if sha(model / 'model.pt') != config['model_sha'] or sha(model / 'model.onnx') != config['onnx_sha']:
        raise ValueError('frozen selected model checksum mismatch')
    if config['onnx_data_sha'] and sha(model / 'model.onnx.data') != config['onnx_data_sha']:
        raise ValueError('frozen ONNX external data checksum mismatch')
    # Register the new managed run in the parent's read-only dashboard; parent training is unchanged.
    (source / 'final-test.json').write_text(json.dumps(dict(run=str(out.parent), out=str(out)), indent=2))
    learned = play(out, 'learned', starts, model, worker, 2000, a.workers)
    teacher = play(out, 'mcts', starts, None, worker, 20000, a.workers, teacher=True)
    result = score(rows, teacher, learned)
    result['mcts_confidence_95'] = wilson(round(result['mcts_win_rate'] * result['n']), result['n'])
    result.update(final_test=True, n_reserved=600, selected_model=config['selected_model'],
                  learned_summary=json.loads((learned / 'summary.json').read_text()),
                  mcts_summary=json.loads((teacher / 'summary.json').read_text()))
    (out / 'summary.json').write_text(json.dumps(result, indent=2))
    lo, hi = result['confidence_95']
    ml, mh = result['mcts_confidence_95']
    gap_se = result['gap_se']
    lines = ['# Reserved 600-seed final test', '',
             'Preselected update-5 specialist, fixed Barricade deck, 41/75 HP, fixed relics/counters, no potions.',
             'Fresh network trained with 200 teacher bootstrap + 500 learner fights. No training or checkpoint selection on these final seeds.', '',
             '| agent | paired wins | win rate | 95% Wilson interval |', '|---|---:|---:|---:|',
             f'| Learned2k | {result["wins"]}/{result["n"]} | {100*result["win_rate"]:.2f}% | {100*lo:.2f}–{100*hi:.2f}% |',
             f'| MCTS20k | {round(result["mcts_win_rate"]*result["n"])}/{result["n"]} | {100*result["mcts_win_rate"]:.2f}% | {100*ml:.2f}–{100*mh:.2f}% |', '',
             f'Paired learned-minus-MCTS gap: {100*result["gap"]:+.2f} points; SE {100*gap_se:.2f} points.' if gap_se is not None else 'Paired gap SE unavailable.',
             f'Paired coverage: {result["n"]}/600 starts. Status counts are recorded in summary.json; caps/errors are not invented as outcome labels.', '',
             'Results are conditional on this one fixed task, not general deck superiority or proof that all seeds are winnable.',
             'Search budgets and objectives differ: learned search is win-only; rollout MCTS also values HP/resources.',
             'Per-agent decision_seconds is recorded for timing context; timing uncertainty has not been quantified.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
