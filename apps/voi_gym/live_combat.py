from contextlib import contextmanager
from math import isfinite
import subprocess
from threading import Timer

from agents.combat.search.voi_gym.model import Belief, adaptive_belief
from apps.voi_combat.benchmark import exchange

from .record_combat import action_labels, record_decision


def settings(data):
    price = float(data['price'])
    budget = int(data['budget'])
    shortlist = int(data['shortlist'])
    probe_sd = float(data['probe_sd'])
    prior_scale = float(data['prior_scale'])
    cap = float(data['initial_sd_cap'])
    mode = data['uncertainty']
    if not (isfinite(price) and price >= 0 and 0 <= budget <= 64 and 1 <= shortlist <= 40
            and 0 < probe_sd <= 1 and 0 <= prior_scale <= 1 and 0 <= cap <= .5
            and mode in ('fixed', 'adaptive')):
        raise ValueError('Invalid replay settings')
    return dict(price=price, budget=budget, shortlist=shortlist, probe_sd=probe_sd,
                prior_scale=prior_scale, initial_sd_cap=cap, uncertainty=mode)


def search(proc, root, config):
    options = dict(config)
    factory = adaptive_belief if options.pop('uncertainty') == 'adaptive' else Belief
    return record_decision(proc, root, belief_factory=factory, **options)[:2]


def restore_prefix(proc, start, prefix):
    root = exchange(proc, start)
    for decision in prefix:
        frames = decision['frames']
        initial = frames[0]['state']
        labels = action_labels(root.get('actions', []))
        if (root.get('step') != decision['step'] or root.get('turn') != decision['turn']
                or root.get('value') != initial['root_value'] or root.get('requests') != initial['q_calls']
                or set(labels) != set(initial['prior'])):
            raise ValueError('Recording does not match the live model or fight start')
        for previous, frame in zip(frames, frames[1:]):
            obs = frame['observation']
            result = exchange(proc, {'pair': [b['action'] for b in obs['branches']]})
            if (result['difference'] != obs['observation'] or
                    result['requests'] != frame['state']['q_calls'] - previous['state']['q_calls']):
                raise ValueError('Recorded probe does not match the live model')
        root = exchange(proc, {'play': labels.index(frames[-1]['state']['move'])})
    return root


def play_branch(proc, start, decisions, origin, history, config, operation):
    root = restore_prefix(proc, start, decisions[:origin])
    target = decisions[origin]
    if (root.get('step') != target['step'] or root.get('turn') != target['turn']
            or root.get('combat') != target.get('combat')
            or root.get('value') != target['frames'][0]['state']['root_value']
            or root.get('requests') != target['frames'][0]['state']['q_calls']):
        raise ValueError('Could not restore the selected recorded position')
    for committed in history:
        if 'win' in root:
            raise ValueError('Fight already ended')
        _, move = search(proc, root, committed)
        root = exchange(proc, {'play': move})
    if 'win' in root:
        raise ValueError('Fight already ended')
    turn = root['turn']
    replay, move = search(proc, root, config)
    played = []
    while operation != 'rerun':
        played.append(replay['frames'][-1]['state']['move'])
        history.append(config)
        root = exchange(proc, {'play': move})
        if 'win' in root:
            return dict(native=True, live=True, history=history, terminal=root, played=played, **replay)
        replay, move = search(proc, root, config)
        if operation == 'action' or root['turn'] != turn:
            break
    return dict(native=True, live=True, history=history, played=played, **replay)


@contextmanager
def bridge_process(command):
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as proc:
        timer = Timer(60, proc.kill)
        timer.start()
        try:
            yield proc
        finally:
            timer.cancel()
            # The bridge normally waits for the next probe or move at this point.
            if proc.poll() is None:
                proc.terminate()
