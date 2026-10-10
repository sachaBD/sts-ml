"""Read-only single-deck experiment dashboard. No jobs started/stopped; safe while running.

  python -m apps.run_rl.single_deck_status --run runs/schema=combat_v4/.../id=...
  watch -n 5 '.venv/bin/python -m apps.run_rl.single_deck_status --run ...'
"""
import argparse
import json
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def journal_counts(directory):
    counts = dict(logged=0, completed=0, wins=0, capped=0, error=0)
    try:
        with (directory / 'results.jsonl').open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a writer may be in the middle of its current line
                counts['logged'] += 1
                status = row.get('status')
                if status in counts:
                    counts[status] += 1
                if status == 'completed':
                    counts['wins'] += bool(row.get('fight', {}).get('won'))
    except FileNotFoundError:
        pass
    return counts


def training_progress(directory, epochs):
    records = []
    try:
        for line in (directory / 'logs/train.log').read_text().splitlines():
            if line.startswith('{"epoch"'):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    if (directory / 'train.done').exists():
        return f'{len(records)}/{epochs} epochs; DONE'
    if records:
        return f'{len(records)}/{epochs} epochs logged; training/export unfinished'
    return f'0/{epochs} epochs logged'


def stages(out, config):
    sequence = [('Bootstrap teacher play', out, 'bootstrap-play', config['bootstrap']),
                ('Encode bootstrap', out, 'bootstrap-rows', None),
                ('MCTS monitoring reference', out, 'reference-monitor', config['monitor']),
                ('Encode monitoring labels', out, 'monitor-rows', None),
                ('Train fresh bootstrap model', out / 'bootstrap', 'train', None),
                ('Evaluate bootstrap model', out / 'bootstrap', 'eval', config['monitor'])]
    for i in range(1, config['updates'] + 1):
        d = out / f'iter{i:03d}'
        sequence += [(f'Update {i}: learner collection', d, 'collect', config['batch_fights']),
                     (f'Update {i}: encode', d, 'rows', None),
                     (f'Update {i}: retrain', d, 'train', None)]
        if i % config['eval_every'] == 0 or i == config['updates']:
            sequence.append((f'Update {i}: evaluate', d, 'eval', config['monitor']))
    return sequence


def render(run):
    metadata = read_json(run / 'run.json') or {}
    out = run / 'out'
    config = read_json(out / 'config.json')
    if config is None:
        return f'{run.name}: controller not configured yet\nManaged state: {metadata.get("status", "unknown")}'
    sequence = stages(out, config)
    pending = [s for s in sequence if not (s[1] / (s[2] + '.done')).exists()]
    active = pending[0] if pending else None
    current = active[0] if active else 'All scheduled stages complete'
    native_workers = config['workers']
    state = metadata.get('status', 'unknown').upper()
    final_link = read_json(out / 'final-test.json')
    final_run = Path(final_link['run']) if final_link else None
    final_out = Path(final_link['out']) if final_link else None
    final_meta = read_json(final_run / 'run.json') if final_run else None
    if final_meta:
        state = 'TRAINING ' + state + ' | FINAL TEST ' + final_meta.get('status', 'unknown').upper()
        if not (final_out / 'learned.done').exists():
            current = 'Final test: learned2k, 600 seeds'
        elif not (final_out / 'mcts.done').exists():
            current = 'Final test: MCTS20k, same 600 seeds'
        else:
            current = 'Training and final test stages complete'
    lines = [f'RUN: {run.name.removeprefix("id=")}   {state}',
             'Controller: apps.run_rl.single_deck (one sequential pipeline)',
             f'CURRENT: {current}',
             f'Workers: {native_workers}   Planned: {config["bootstrap"]} teacher fights + '
             f'{config["updates"]} × {config["batch_fights"]} learner fights',
             f'Pipeline stages complete: {len(sequence)-len(pending)}/{len(sequence)}', '']
    if metadata.get('status') == 'failed':
        lines += [f'FAILURE: inspect {run / "logs/stderr.log"}', '']
    def play_line(label, directory, target):
        c = journal_counts(directory)
        text = f'{label}: {c["logged"]}/{target} results; {c["wins"]} wins'
        if c['capped'] or c['error']:
            text += f'; {c["capped"]} capped, {c["error"]} errors'
        return text
    lines += [play_line('Bootstrap MCTS data', out / 'bootstrap-play', config['bootstrap']),
              play_line('MCTS reference', out / 'reference-monitor', config['monitor']),
              'Bootstrap training: ' + training_progress(out / 'bootstrap', config['bootstrap_epochs']),
              play_line('Bootstrap evaluation', out / 'bootstrap/eval', config['monitor']), '']
    collected = 0
    for i in range(1, config['updates'] + 1):
        d = out / f'iter{i:03d}'
        c = journal_counts(d / 'collect')
        collected += c['logged']
        if not (d / 'collect/results.jsonl').exists():
            lines.append(f'Update {i}: not started')
            continue
        line = play_line(f'Update {i} collection', d / 'collect', config['batch_fights'])
        if (d / 'logs/train.log').exists():
            line += '; train ' + training_progress(d, config['epochs'])
        if (d / 'eval/results.jsonl').exists():
            line += f'; evaluation {journal_counts(d / "eval")["logged"]}/{config["monitor"]}'
        lines.append(line)
    lines += ['', f'Learner collection total: {collected}/{config["updates"]*config["batch_fights"]} fights']
    curves = read_json(out / 'curve.json') or []
    if curves:
        latest = curves[-1]
        interval = latest['confidence_95']
        lines += [f'Latest monitoring result: update {latest["iteration"]}, '
                  f'{latest["wins"]}/{latest["n"]} wins ({100*latest["win_rate"]:.1f}%; '
                  f'95% Wilson {100*interval[0]:.1f}–{100*interval[1]:.1f}%)',
                  f'MCTS reference on those paired seeds: {100*latest["mcts_win_rate"]:.1f}%',
                  f'Plot: {out / "curve.png"}']
    if (out/'training-win-rate.png').exists():
        lines.append(f'Training-data win-rate plot (updates each round): {out / "training-win-rate.png"}')
    if final_out:
        final_config = read_json(final_out / 'config.json') or {}
        lines += ['', f'FINAL 600-SEED TEST: {final_meta.get("status", "unknown").upper() if final_meta else "REGISTERED"}; workers {final_config.get("workers", "unknown")}',
                  play_line('Final learned2k', final_out / 'learned', 600),
                  play_line('Final MCTS20k', final_out / 'mcts', 600)]
        result = read_json(final_out / 'summary.json')
        if result:
            lo, hi = result['confidence_95']
            lines += [f'Final paired result: {result["wins"]}/{result["n"]} learned wins '
                      f'({100*result["win_rate"]:.2f}%; 95% Wilson {100*lo:.2f}–{100*hi:.2f}%)',
                      f'Final MCTS: {100*result["mcts_win_rate"]:.2f}%; paired gap {100*result["gap"]:+.2f} points',
                      f'Final report: {final_out / "REPORT.md"}']
        else:
            phase = 'learned' if not (final_out / 'learned.done').exists() else 'mcts'
            lines.append(f'Current final log: {final_out / "logs" / (phase + ".log")}')
        lines.append(f'Final stage log: {final_run / "logs/stdout.log"}')
    else:
        lines += ['', 'Final 600-seed test: NOT RUN (reserved)']
    if active:
        lines.append(f'Current detailed log: {active[1] / "logs" / (active[2]+".log")}')
    lines.append(f'Stage log: {run / "logs/stdout.log"}')
    lines.append('Counts are persisted results; some in-flight work may not be logged yet.')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    print(render(a.run))


if __name__ == '__main__':
    main()
