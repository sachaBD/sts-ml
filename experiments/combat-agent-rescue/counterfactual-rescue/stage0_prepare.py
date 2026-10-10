#!/usr/bin/env python3
"""Freeze counterfactual-rescue Stage-1 starts and audit restart public history.

No continuation is played: the frozen champ_session is used only for replayed public
views.  The source teacher JSONL and its successful-record membership are immutable
inputs; selection is deterministic from those already-recorded outcomes.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path('runs/schema=combat_v4/date=2026-10-06')
SOURCE = ROOT / 'id=demon-form-teacher-correction-v1/out/teacher-play/results.jsonl'
FROZEN_SESSION = ROOT / 'id=demon-form-continuation-v1/out/frozen/champ_session'
OUT = ROOT / 'id=counterfactual-rescue-v1/out'
SELECTION_TAG = 'counterfactual-rescue-v1:successful-teacher-path-id'
END_TURN = 2147483648


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chosen_restart_prefixes(actions):
    """Restart at nearest post-end-turn boundaries to 25%, 50%, 75% player-turn progress.

    A terminal action is never a restart.  Repeated target boundaries are rejected,
    rather than silently turning a short fight into duplicate restart states.
    """
    ends = [i for i, a in enumerate(actions) if a == END_TURN]
    if len(ends) < 3:
        return None
    ranks, used = [], set()
    for label, fraction in [('early', .25), ('middle', .50), ('late', .75)]:
        target = fraction * len(ends)
        rank = min(range(1, len(ends) + 1), key=lambda k: (abs(k - target), k))
        if rank in used or ends[rank - 1] + 1 >= len(actions):
            return None
        used.add(rank)
        ranks.append((label, rank, ends[rank - 1] + 1))
    return ranks


def freeze(out):
    dest = out / 'starts.json'
    if dest.exists():
        raise FileExistsError(f'refusing to overwrite frozen {dest}')
    records = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    wins = [r for r in records if r['status'] == 'completed' and r['fight']['won']]
    ranked = sorted(wins, key=lambda r: hashlib.sha256((SELECTION_TAG + ':' + r['fight_id']).encode()).hexdigest())
    selected = ranked[:12]
    rows = []
    for rank, r in enumerate(selected, 1):
        fight = r['fight']
        restarts = chosen_restart_prefixes(fight['actions'])
        if restarts is None:
            raise ValueError(f'{fight["fight_id"]}: cannot make 3 unique nonterminal turn-boundary restarts')
        rows.append({
            'selection_rank': rank,
            'selection_hash': hashlib.sha256((SELECTION_TAG + ':' + fight['fight_id']).encode()).hexdigest(),
            'fight_id': fight['fight_id'], 'start': fight['start'], 'teacher_actions': fight['actions'],
            'teacher_end_turn_steps': [i for i, a in enumerate(fight['actions']) if a == END_TURN],
            'restarts': [{'label': label, 'completed_player_turns': turn, 'prefix_actions': actions[:step],
                          'prefix_length': step, 'progress': turn / len([a for a in actions if a == END_TURN])}
                         for label, turn, step in restarts for actions in [fight['actions']]],
        })
    payload = {'stage': 0, 'selection_tag': SELECTION_TAG, 'source': str(SOURCE), 'source_sha256': sha(SOURCE),
               'source_records': len(records), 'successful_teacher_paths': len(wins), 'selected_starts': len(rows),
               'restart_states_requested': len(rows) * 3,
               'selection_rule': 'ascending sha256(selection_tag + ":" + fight_id), first 12 successful teacher paths',
               'restart_rule': 'nearest unique post-end-turn boundaries to 25%, 50%, 75% of recorded player turns',
               'rows': rows}
    dest.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'starts': len(rows), 'restart_states': len(rows) * 3, 'sha256': sha(dest)}, indent=2))


def infer_known_prefix(views):
    # Same deck-specific history helper logic as single-deck continuation.py.
    k, ks, anomalies = 0, [0], []
    for j in range(len(views) - 1):
        a, b = views[j], views[j + 1]
        delta = len(b['draw']) - len(a['draw'])
        if a['kind'] == 'headbutt':
            if delta != 1:
                anomalies.append([j, 'headbutt draw delta', delta])
            k += 1
        elif delta < 0:
            k = max(0, k + delta)
        elif delta > 0:
            k = 0
        ks.append(k)
    return ks, anomalies


def audit(out):
    starts = json.loads((out / 'starts.json').read_text())
    if sha(SOURCE) != starts['source_sha256']:
        raise ValueError('teacher source changed since start freeze')
    dest = out / 'restart-history-audit.json'
    if dest.exists():
        raise FileExistsError(f'refusing to overwrite {dest}')
    requests, keys = [], []
    for row in starts['rows']:
        for restart in row['restarts']:
            for n in range(restart['prefix_length'] + 1):
                requests.append({'start': row['start'], 'ops': [{'act': a} for a in restart['prefix_actions'][:n]], 'query': 'view'})
            keys.append((row, restart, len(requests) - restart['prefix_length'] - 1, len(requests)))
    proc = subprocess.run([str(FROZEN_SESSION)], input=''.join(json.dumps(r) + '\n' for r in requests),
                          text=True, capture_output=True, check=True, timeout=60)
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    if len(replies) != len(requests):
        raise ValueError('frozen session reply count mismatch')
    rows = []
    for row, restart, lo, hi in keys:
        views = [x['view'] for x in replies[lo:hi]]
        ks, anomalies = infer_known_prefix(views)
        view = views[-1]
        rows.append({'fight_id': row['fight_id'], 'label': restart['label'], 'prefix_length': restart['prefix_length'],
                     'completed_player_turns': restart['completed_player_turns'], 'known_prefix': ks[-1],
                     'any_known_before': any(ks), 'anomalies': anomalies, 'terminal': view['kind'] != 'play',
                     'legal_actions': [m['bits'] for m in view['legal']]})
    payload = {'starts_sha256': sha(out / 'starts.json'), 'session': str(FROZEN_SESSION), 'session_sha256': sha(FROZEN_SESSION),
               'states': rows, 'summary': {'requested': len(rows), 'valid_k0_nonterminal': sum(not x['known_prefix'] and not x['anomalies'] and not x['terminal'] for x in rows),
               'known_order_excluded': sum(x['known_prefix'] > 0 for x in rows), 'audit_anomaly_excluded': sum(bool(x['anomalies']) for x in rows),
               'terminal_excluded': sum(x['terminal'] for x in rows)}}
    dest.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload['summary'], indent=2))


p = argparse.ArgumentParser()
p.add_argument('command', choices=['freeze', 'audit'])
p.add_argument('--out', type=Path, default=OUT)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
{'freeze': freeze, 'audit': audit}[a.command](a.out)
