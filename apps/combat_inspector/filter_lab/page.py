"""Turn filter_replay records into filter-lab decisions and render the page."""
from pathlib import Path
import json, re

here = Path(__file__).resolve().parent
BUDGETS = [8, 24, 64, 160]
RECORDED = {'price': 0.0005, 'cap': 160, 'worlds': 16, 'mode': 'full', 'explore': 0, 'seed': 0}


def title(s):
    return s.replace('_', ' ').title()


def clean(desc):
    s = desc.strip('{} ')
    s = re.sub(r'use card \(\d+\) \(([^,]+),[^)]*\)', r'\1', s)
    s = re.sub(r'drink potion \(\d+\) ', 'Drink ', s)
    s = re.sub(r' -> \((\d+)\) ([A-Z_]+)', lambda m: ' → ' + title(m[2].lower()) + ' #' + str(int(m[1]) + 1), s)
    return 'End turn' if s == 'end turn' else s


def card(c):
    return {'name': c['name'] + ('+' if c['upgraded'] else ''), 'cost': c['cost_for_turn'], 'type': c['type']}


def enemy(i, m):
    e = {'name': f"{title(m['id'])} #{i + 1}", 'hp': m['hp'], 'max': m['max_hp'], 'block': m['block'],
         'statuses': m['statuses'], 'intent': None}
    if m['intent']:
        name = title(m['intent'].lower().removeprefix(m['id'] + '_'))
        if 'intent_damage' in m:
            hits = m.get('intent_hits', 1)
            name += f" · attacks {m['intent_damage']}" + (f" × {hits}" if hits > 1 else '')
        e['intent'] = name
    return e


def decision(r, settings=RECORDED):
    v = r['view']
    names = [clean(m['desc']) for m in v['legal']]
    s0 = r['scores']
    prior = max(range(len(s0)), key=s0.__getitem__)
    chosen = next(i for i, m in enumerate(v['legal']) if m['bits'] == r['chosen'])
    trials = []
    for t in r['trials']:
        line = [clean(x) for x in t['line']]
        assert line[0] == names[t['arm']]
        trials.append({k: t[k] for k in ('arm', 'world', 'spent', 'interior', 'rate', 'candidate', 'reference',
                                          'before', 'after', 'sd0', 'sd1', 'scores')}
                      | {'line': line, 'ref_line': [clean(x) for x in t['reference_line']]})
    final = trials[-1]['scores'] if trials else s0
    leader = max(range(len(final)), key=final.__getitem__)
    assert settings['explore'] > 0 or leader == chosen
    assert all(b['spent'] <= t['spent'] for b, t in zip(trials, trials[1:]))
    return {
        'fight': r['fight'], 'step': r['step'], 'turn': v['turn'] + 1, 'settings': settings,
        'player': v['player'], 'hand': [card(c) for c in v['hand']], 'potions': v['potions'],
        'draw': len(v['draw']), 'discard': len(v['discard']),
        'enemies': [enemy(i, m) for i, m in enumerate(v['monsters']) if m['alive']],
        'moves': names, 'prior': prior, 'chosen': chosen, 'leader': leader, 'scores': s0,
        'opening': r['opening'], 'value': r['value'], 'root_cached': r['root_cached'],
        'reserve': r['reserve'], 'spent': r['spent'], 'stop': r['stop'], 'last_rate': r['last_rate'],
        'next': {'turn': r['next_turn'] + 1, 'hp': r['next_hp'], 'energy': r['next_energy']},
        'trials': trials}


def page(records):
    fights, order = {}, []
    for r in records:
        fid = r['fight']
        if fid not in fights:
            boss = 'The Champ' if 'champ' in fid else 'Bronze Automaton'
            fights[fid] = {'label': f"{boss} · deck {fid.split('mc-')[1][:6]}", 'decisions': []}
            order.append(fid)
        fights[fid]['decisions'].append(decision(r))
    data = {'price': RECORDED['price'], 'cap': RECORDED['cap'], 'worlds': RECORDED['worlds'], 'budgets': BUDGETS,
            'fights': [fights[f] for f in order]}
    blob = json.dumps(data, allow_nan=False, separators=(',', ':')).replace('<', '\\u003c')
    return (here / 'template.html').read_text().replace('@@DATA@@', blob)


def load_records(trace):
    return [json.loads(l) for l in Path(trace).read_text().splitlines()]
