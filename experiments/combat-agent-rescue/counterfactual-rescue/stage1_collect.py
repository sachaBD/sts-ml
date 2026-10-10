#!/usr/bin/env python3
"""Frozen-input dispatcher for approved Stage-1 continuations; collection is explicit."""
import argparse, concurrent.futures, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path('runs/schema=combat_v4/date=2026-10-06/id=counterfactual-rescue-v1')
OUT, FROZEN = ROOT / 'out', ROOT / 'out/frozen'
BIN, MODEL = FROZEN / 'pv_continuation', FROZEN / 'model/model.onnx'
MODEL_DATA, HASHES = FROZEN / 'model/model.onnx.data', FROZEN / 'stage1-input-sha256.txt'
ACTIVE = set()


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expected_hashes():
    return {Path(line.split(maxsplit=1)[1]).name: line.split()[0] for line in HASHES.read_text().splitlines()}


def verify_frozen():
    wanted = expected_hashes()
    files = [BIN, MODEL, MODEL_DATA, FROZEN / 'src/stage1_collect.py', FROZEN / 'src/continuation.cpp']
    for file in files:
        if not file.is_file() or wanted.get(file.name) != sha(file):
            raise ValueError(f'frozen input missing or hash mismatch: {file}')
    if sha(Path(__file__)) != wanted['stage1_collect.py']:
        raise ValueError('dispatcher must be run from its frozen source copy')


def load_jobs():
    starts, audit = json.loads((OUT / 'starts.json').read_text()), json.loads((OUT / 'restart-history-audit.json').read_text())
    if starts['selected_starts'] != 12 or starts['restart_states_requested'] != 36 or sha(OUT / 'starts.json') != audit['starts_sha256']:
        raise ValueError('frozen start manifest mismatch')
    good = {(x['fight_id'], x['label']) for x in audit['states'] if x['known_prefix'] == 0 and not x['anomalies'] and not x['terminal']}
    if len(audit['states']) != 36 or len(good) != 36: raise ValueError('need exactly 36 eligible audited restart states')
    jobs = []
    for row in starts['rows']:
        for restart in row['restarts']:
            if (row['fight_id'], restart['label']) not in good: raise ValueError('manifest restart silently filtered')
            for particle in range(4):
                for mode, sims in [('learner', 2000), ('teacher', 20000)]:
                    jobs.append({'fight_id': row['fight_id'], 'position': restart['label'], 'prefix_length': restart['prefix_length'],
                                 'particle': particle, 'mode': mode, 'sims': sims, 'start': row['start'], 'ops': restart['prefix_actions'],
                                 'particles': [particle], 'true_state': False, 'record': mode == 'learner'})
    keys = [(j['fight_id'], j['position'], j['particle'], j['mode']) for j in jobs]
    if len(jobs) != 288 or len(keys) != len(set(keys)): raise ValueError(f'expected 288 unique jobs, got {len(jobs)}')
    return jobs


def kill_group(proc):
    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError: pass


def execute(job):
    command = [str(BIN)] + ([str(MODEL)] if job['mode'] == 'learner' else [])
    started = time.time(); started_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started)); proc = None
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        ACTIVE.add(proc)
        stdout, stderr = proc.communicate(json.dumps(job) + '\n', timeout=300)
        response = json.loads(stdout) if proc.returncode == 0 and stdout.count('\n') == 1 else None
        inner = response.get('results', [{}])[0] if response and len(response.get('results', [])) == 1 else {}
        status = inner.get('status', 'error') if response and not response.get('error') else 'error'
        if status not in ('completed', 'capped'): status = 'error'
        return {'job': {k: v for k, v in job.items() if k not in ('start', 'ops')}, 'status': status, 'started_at': started_iso,
                'finished_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seconds': time.time()-started,
                'response': response, 'returncode': proc.returncode, 'stderr': stderr[-2000:]}
    except subprocess.TimeoutExpired:
        if proc: kill_group(proc)
        return {'job': {k: v for k, v in job.items() if k not in ('start', 'ops')}, 'status': 'error', 'started_at': started_iso,
                'finished_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seconds': time.time()-started, 'error': '300s timeout'}
    finally:
        if proc: ACTIVE.discard(proc)


def cleanup_interrupt(*_):
    for proc in list(ACTIVE): kill_group(proc)
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['plan', 'collect', 'selftest-cleanup']); ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    if a.command == 'selftest-cleanup':
        p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True); kill_group(p); p.wait(timeout=3)
        print(json.dumps({'dummy_child_cleanup': p.returncode is not None})); return 0 if p.returncode is not None else 1
    if not 1 <= a.workers <= 10: raise ValueError('workers must be 1..10')
    verify_frozen(); jobs = load_jobs()
    config = {'starts_sha256': sha(OUT/'starts.json'), 'history_audit_sha256': sha(OUT/'restart-history-audit.json'), 'binary_sha256': sha(BIN),
              'model_sha256': sha(MODEL), 'model_data_sha256': sha(MODEL_DATA), 'dispatcher_sha256': sha(Path(__file__)), 'jobs': 288, 'workers': a.workers,
              'semantics': 'same public-particle index; teacher salt index+1; learner unsalted; no retries/resume'}
    print(json.dumps(config, indent=2), flush=True)
    if a.command == 'plan': return
    dest, cfg, summary = OUT/'stage1-results.jsonl', OUT/'stage1-config.json', OUT/'stage1-summary.json'
    if any(x.exists() for x in (dest, cfg, summary)): raise FileExistsError('refusing retry/resume/cache merge')
    cfg.write_text(json.dumps(config, indent=2)+'\n')
    signal.signal(signal.SIGINT, cleanup_interrupt); signal.signal(signal.SIGTERM, cleanup_interrupt)
    counts = {'completed': 0, 'capped': 0, 'error': 0}
    with dest.open('w') as f, concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(execute, j) for j in jobs]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result(); counts[result['status']] += 1; f.write(json.dumps(result)+'\n'); f.flush(); os.fsync(f.fileno())
            print(f'{n}/288 {result["job"]["mode"]} {result["status"]} {result["seconds"]:.1f}s counts={counts}', flush=True)
    summary.write_text(json.dumps({'attempted': 288, 'statuses': counts, 'exit': 0 if not counts['error'] else 2}, indent=2)+'\n')
    if counts['error']: raise SystemExit(2)

if __name__ == '__main__': main()
