"""Turn filter_replay records into filter-lab decisions and render the page."""
from pathlib import Path
import json, math, re

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


def node(n):
    logits = n['logits']
    if logits:
        top = max(logits)
        e = [math.exp(x - top) for x in logits]
        z = sum(e)
        policy = [round(x / z, 4) for x in e]
    else:
        policy = []
    return {
        'id': n['id'], 'parent': n['parent'], 'parent_edge': n['parent_edge'], 'world': n['world'],
        'born': n['born'], 'turn': n['turn'] + 1, 'key': n['key'],
        'terminal': n['terminal'], 'cycle': n['cycle'], 'cached': n['cached'],
        'nn_value': round(n['nn_value'], 5), 'prior_sd': round(n['prior_sd'], 5),
        'logits': [round(x, 4) for x in logits], 'policy': policy,
        'thinking': [round(x, 5) for x in n['thinking']],
        'actions': [clean(a) for a in n['actions']],
        'edge_chance': n['edge_chance'],
    }


def step_rec(s):
    return {
        'parent': s['parent'], 'edge': s['edge'], 'world': s['world'], 'child': s['child'],
        'spent': s['spent'], 'world_sample': s['world_sample'],
        'voi': round(s['voi'], 6), 'weight': round(s['weight'], 5),
        'gap': round(s['gap'], 5), 'innovation': round(s['innovation'], 5),
        'root_q': [round(x, 5) for x in s['root_q']],
    }


def decision(r, settings=RECORDED):
    v = r['view']
    names = [clean(m['desc']) for m in v['legal']]
    nodes = [node(n) for n in r['nodes']]
    steps = [step_rec(s) for s in r['steps']]
    q0 = [round(x, 5) for x in r['initial_q']]
    root_logits = nodes[0]['logits']
    prior = max(range(len(root_logits)), key=root_logits.__getitem__)
    chosen = next(i for i, m in enumerate(v['legal']) if m['bits'] == r['chosen'])
    final_q = steps[-1]['root_q'] if steps else q0
    raw_final_q = r['steps'][-1]['root_q'] if r['steps'] else r['initial_q']
    leader = max(range(len(raw_final_q)), key=lambda i: (raw_final_q[i], root_logits[i]))
    assert settings['explore'] > 0 or leader == chosen
    assert all(b['spent'] <= t['spent'] for b, t in zip(steps, steps[1:]))
    max_turn = max(n['turn'] for n in nodes)
    return {
        'fight': r['fight'], 'step': r['step'], 'turn': v['turn'] + 1, 'max_turn': max_turn, 'settings': settings,
        'player': v['player'], 'hand': [card(c) for c in v['hand']], 'potions': v['potions'],
        'draw': len(v['draw']), 'discard': len(v['discard']),
        'enemies': [enemy(i, m) for i, m in enumerate(v['monsters']) if m['alive']],
        'moves': names, 'prior': prior, 'chosen': chosen, 'leader': leader, 'initial_q': q0,
        'opening': [round(x, 5) for x in r['opening']], 'value': round(r['value'], 5), 'root_cached': r['root_cached'],
        'spent': r['spent'], 'stop': r['stop'], 'last_voi': round(r['last_voi'], 6),
        'next': {'turn': r['next_turn'] + 1, 'hp': r['next_hp'], 'energy': r['next_energy']},
        'nodes': nodes, 'steps': steps,
    }


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
