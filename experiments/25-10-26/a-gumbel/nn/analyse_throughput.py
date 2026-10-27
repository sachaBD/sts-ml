"""Summarize durable per-trial results, telemetry and GPU progress. No gameplay."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import statistics


def telemetry(directory, result):
    rows = []
    path = directory / 'gpu.csv'
    if not path.exists():
        return {}
    for row in csv.DictReader(path.open()):
        row = {k.strip(): v.strip() for k,v in row.items()}
        try:
            stamp = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').timestamp()
            # Approximate phase alignment: process launch + native setup. Input parsing/startup add small offsets.
            begin = result['wall_begin'] + result['setup_seconds'] + .2
            if begin <= stamp <= begin + result['gameplay_seconds']:
                rows.append(row)
        except (ValueError, KeyError):
            continue
    data = dict(samples=len(rows), sampling_ms=500, alignment='approximate gameplay interval')
    for label, field in [('gpu_util_pct','utilization.gpu [%]'), ('memory_util_pct','utilization.memory [%]'),
                         ('power_w','power.draw [W]'), ('clock_mhz','clocks.current.sm [MHz]')]:
        values = []
        for row in rows:
            try: values.append(float(row[field]))
            except (ValueError, KeyError): pass
        if values:
            data[label] = dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values))
    return data


def phases(directory, result):
    points = [json.loads(s) for s in re.findall(r'gpu_stats: (\{[^\n]+\})', (directory/'worker.log').read_text())]
    if not points:
        return {}
    data = {}
    for percent in (50, 90, 99):
        # GPU stats time starts immediately before workers start; completion milestones include evaluator setup.
        boundary = result['milestones'][str(percent)] - result['setup_seconds']
        before = [p for p in points if p['seconds'] <= boundary]
        if not before:
            continue
        a, b = before[-1], points[-1]
        seconds = b['seconds'] - a['seconds']
        leaves, capacity = b['leaves']-a['leaves'], b['capacity']-a['capacity']
        data[f'after_{percent}_pct_games_approx'] = dict(seconds=seconds, leaves=leaves,
            useful_evals_per_second=leaves/seconds if seconds else None,
            fill=leaves/capacity if capacity else None)
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('runs/nn-throughput-20261025'))
    p.add_argument('--tag', default='')
    args = p.parse_args()
    trials = []
    for path in sorted((args.out/'trials').glob('*/summary.json')):
        result = json.loads(path.read_text())
        if args.tag and not result['name'].startswith(args.tag+'-'):
            continue
        timing = (path.parent / 'time.txt').read_text()
        match = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)', timing)
        if match:
            seconds = 0.
            for part in match.group(1).split(':'):
                seconds = seconds * 60 + float(part)
            result['native_wall_seconds'] = seconds
            cpu = sum(float(re.search(p, timing)[1]) for p in
                      (r'User time \(seconds\): ([\d.]+)', r'System time \(seconds\): ([\d.]+)'))
            result['cpu_logical_cores'] = cpu / seconds
        result['telemetry'] = telemetry(path.parent, result)
        result['phases'] = phases(path.parent, result)
        trials.append(result)
    groups = {}
    for r in trials:
        defaults = dict(wait_us=1000, tokens=48)
        key = json.dumps({k:r.get(k, defaults.get(k)) for k in
                          ('batch','tokens','groups','scheduler','parallel','adaptive','wait_us','games')}, sort_keys=True)
        groups.setdefault(key, []).append(r)
    aggregates = []
    metrics = ['gameplay_seconds','elapsed_seconds','useful_evals_per_second','games_per_hour','batch_fill',
               'cpu_logical_cores','evaluator_busy_fraction']
    for key, rs in groups.items():
        aggregate = dict(config=json.loads(key), trials=len(rs))
        for metric in metrics:
            values=[r[metric] for r in rs]
            aggregate[metric]=dict(median=statistics.median(values), minimum=min(values), maximum=max(values))
        tails=[r['tail_seconds']['90'] for r in rs]
        aggregate['last_10pct_seconds']=dict(median=statistics.median(tails),minimum=min(tails),maximum=max(tails))
        aggregates.append(aggregate)
    aggregates.sort(key=lambda r:r['useful_evals_per_second']['median'],reverse=True)
    result=dict(trials=trials,aggregates=aggregates,
                caveat='Ranges are observed variation across listed trials, not confidence intervals. GPU utilization is active-time, not fraction of peak compute.')
    destination=args.out/f'analysis{("-"+args.tag) if args.tag else ""}.json'
    destination.write_text(json.dumps(result,indent=2))
    for a in aggregates:
        c=a['config']
        print(f"b{c['batch']} t{c['tokens']} groups{c['groups']} {c['scheduler']} bank={c['adaptive'] or '-'} wait={c['wait_us']} "
              f"games={c['games']} n={a['trials']}: "
              f"{a['useful_evals_per_second']['median']/1000:.1f}k eval/s, "
              f"{a['gameplay_seconds']['median']:.2f}s gameplay, "
              f"tail10% {a['last_10pct_seconds']['median']:.2f}s, "
              f"fill {a['batch_fill']['median']:.0%}, cores {a['cpu_logical_cores']['median']:.2f}")
    print('saved',destination)


if __name__=='__main__':
    main()
