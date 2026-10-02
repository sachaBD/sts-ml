"""Versioned v3 observation encoder. Legacy records are deliberately rejected, not backfilled.
The map is a layered DAG, represented with row-local adjacency, never complete paths.
"""
import numpy as np
import torch

ROWS, LANES = 16, 7
CARD_FEATURES = 14
ROOM_IDS = {'SHOP': 0, 'REST': 1, 'EVENT': 2, 'ELITE': 3, 'MONSTER': 4,
            'TREASURE': 5, 'BOSS': 6, 'BOSS_TREASURE': 7, 'NONE': 8, 'INVALID': 9}
EVENT_KINDS = ('normal', 'shrine', 'one_time')
RELIC_KINDS = ('common', 'uncommon', 'rare', 'shop', 'boss')
# Fixed order is part of the architecture/input contract.
NUMERIC = ('hp', 'max_hp', 'hp_fraction', 'gold', 'floor', 'floor_in_act', 'potion_capacity', 'empty_potion_slots',
           'ruby', 'sapphire', 'emerald', 'card_rarity_factor', 'rare_hallway', 'rare_elite', 'rare_shop',
           'potion_modifier', 'potion_roll_probability', 'potion_obtain_probability',
           'question_base_fight', 'question_base_shop', 'question_base_treasure',
           'question_next_fight', 'question_next_shop', 'question_next_treasure', 'question_next_event',
           'question_after_shop_fight', 'question_after_shop_shop', 'question_after_shop_treasure', 'question_after_shop_event',
           'shop_remove_count', 'shop_remove_cost', 'hallway_count', 'elite_count', 'relic_candidates_exact')


def encode_graph(items, device='cpu'):
    states = [s for s, _ in items]
    for s in states:
        o = s.get('overworld', {})
        if o.get('version') != 1 or o.get('ascension') != 20 or o.get('character') != 0:
            raise ValueError('run_policy_v3 requires new version=1 observations from A20 Ironclad runs; recollect legacy data')
        if 'nodes' not in s['map']:
            raise ValueError('run_policy_v3 requires map.nodes (do not strip graph observations)')
    B = len(states)
    f = np.float32
    C = max(len(s['deck']) for s in states)
    K = max(1, max(len(opts) for _,opts in items))
    card_features = np.zeros((B,C,CARD_FEATURES),f)
    option_features = np.zeros((B,K,CARD_FEATURES),f)
    for i,(s,opts) in enumerate(items):
        for cards,target in ((s['deck'],card_features),(opts,option_features)):
            for j,c in enumerate(cards):
                if 'mechanics' not in c:
                    raise ValueError('run_policy_v3 requires public card mechanics on deck and offered cards')
                m = c['mechanics']
                if not 0 <= m['type'] < 6:
                    raise ValueError('invalid card type')
                target[i,j,m['type']] = 1
                target[i,j,6:] = [m['rarity']/6,m['cost']/4,m['base_damage']/100,
                                  m['innate'],m['ethereal'],m['exhaust'],m['self_retain'],m['x_cost']]
    rooms = np.full((B, ROWS, LANES), 8, np.int64)
    mask = np.zeros((B, ROWS, LANES), f)
    numeric = np.zeros((B, ROWS, LANES, 3), f)
    edge = np.zeros((B, ROWS-1, LANES, LANES), f)
    roots = np.zeros((B, ROWS, LANES), f)
    sc = np.zeros((B, len(NUMERIC)), f)
    act = np.zeros(B, np.int64)
    current_room = np.zeros(B, np.int64)
    enc = np.zeros((B, 64), f)
    elite = np.zeros((B, 64), f)
    history = np.zeros((B, 4), np.int64) # last/penultimate hallway, last elite, currently visible battle
    event_n = max(1, max(sum(len(s['overworld']['events'][k]) for k in EVENT_KINDS) for s in states))
    relic_n = max(1, max(sum(len(s['overworld']['relic_candidates'].get(k, [])) for k in RELIC_KINDS) for s in states))
    eid = np.zeros((B, event_n), np.int64); ekind = eid.copy()
    emask = np.zeros((B, event_n), f); elig = emask.copy()
    rid = np.zeros((B, relic_n), np.int64); rkind = rid.copy()
    rmask = np.zeros((B, relic_n), f); relig = rmask.copy(); rshop = rmask.copy()
    for b, s in enumerate(states):
        o, m = s['overworld'], s['map']
        act[b] = o['act']; current_room[b] = o['current_room']
        if not 1 <= act[b] <= 4:
            raise ValueError('act must be 1..4')
        en = o['encounters']; keys = o['keys']
        mx = max(s['max_hp'], 1)
        values = [s['hp']/100, s['max_hp']/100, s['hp']/mx, s['gold']/300, s['floor']/60,
                  o['floor_in_act']/16, s['potion_capacity']/5, (s['potion_capacity']-len(s['potions']))/5,
                  float(keys['ruby']), float(keys['sapphire']), float(keys['emerald']), o['card_rarity_factor']/40,
                  *[o['card_rarity'][k]['rare'] for k in ('hallway','elite','shop')], o['potion_chance_modifier']/100,
                  o['potion_roll_probability'], o['potion_obtain_probability'],
                  *[o['question_base'][k] for k in ('fight','shop','treasure')],
                  *[o[k][v] for k in ('question_next','question_after_shop') for v in ('fight','shop','treasure','event')],
                  o['shop_remove_count']/10, o['shop_remove_cost']/300,
                  en['hallway_count']/16, en['elite_count']/10, float(o['relic_candidates_exact'])]
        sc[b] = values
        for q in en['next_hallway']:
            enc[b, q['id']] = q['probability']
        for id_ in en['possible_elites']:
            elite[b, id_] = 1 / max(len(en['possible_elites']), 1)
        h = en['hallway_history']
        history[b] = [h[-1] if h else 0, h[-2] if len(h)>1 else 0, en['last_elite'], o['visible_encounter']]
        j = 0
        for kind, key in enumerate(EVENT_KINDS):
            for q in o['events'][key]:
                eid[b,j], ekind[b,j], emask[b,j], elig[b,j] = q['id'], kind, 1, q['eligible']; j += 1
        j = 0
        for kind, key in enumerate(RELIC_KINDS):
            for q in o['relic_candidates'].get(key, []):
                rid[b,j], rkind[b,j], rmask[b,j], relig[b,j], rshop[b,j] = q['id'], kind, 1, q['eligible'], q['shop_eligible']; j += 1
        nodes = {(q['y'],q['x']):q for q in m['nodes'] if q['room'] not in ('NONE','INVALID')}
        if any(not (0 <= y < 15 and 0 <= x < LANES) for y,x in nodes):
            raise ValueError('map node out of range')
        # Standard maps omit the boss node. Add one explicit sink, not a hidden boss schedule.
        if act[b] <= 3:
            nodes[(15,0)] = dict(x=0,y=15,room='BOSS',edges=[])
            for (y,x), q in list(nodes.items()):
                if y == 14:
                    nodes[(y,x)] = {**q, 'edges':[0]}
        cy, cx = m['current']['y'], m['current']['x']
        ny = cy + 1
        if 'next_xs' in m: # committed path after-state: only its selected first edge is reachable
            starts = [(ny,int(x)) for x in m['next_xs']]
        elif cy < 0:
            starts = [(0,x) for (y,x),q in nodes.items() if y == 0 and q['edges']]
        elif (cy,cx) in nodes:
            starts = [(ny,x) for x in nodes[(cy,cx)]['edges']]
        else:
            starts = []
        # At rewards/rest, the current room has been resolved; only future nodes enter the graph.
        reachable, todo = set(), list(starts)
        while todo:
            key = todo.pop()
            if key in reachable:
                continue
            if key not in nodes:
                raise ValueError(f'map edge points to absent node: {key}')
            reachable.add(key)
            y,x = key
            if y < ROWS-1:
                todo.extend((y+1,z) for z in nodes[key]['edges'])
        burning = m.get('burning_elite')
        for y,x in reachable:
            q = nodes[(y,x)]
            rooms[b,y,x] = ROOM_IDS[q['room']]; mask[b,y,x] = 1
            numeric[b,y,x] = [y/15, (y-cy)/16, float(bool(burning and burning['x']==x and burning['y']==y))]
            if y < ROWS-1:
                for z in q['edges']:
                    edge[b,y,x,z] = 1
        for y,x in starts:
            roots[b,y,x] = 1
    arrays = dict(card_mechanics=card_features, opt_mechanics=option_features,
                  graph_room=rooms, graph_mask=mask, graph_numeric=numeric, graph_edge=edge, graph_roots=roots,
                  run_numeric=sc, act=act, current_room=current_room, hallway_probability=enc, elite_probability=elite,
                  encounter_history=history, event_id=eid, event_kind=ekind, event_mask=emask, event_eligible=elig,
                  candidate_relic_id=rid, candidate_relic_kind=rkind, candidate_relic_mask=rmask,
                  candidate_relic_eligible=relig, candidate_relic_shop_eligible=rshop)
    return {k:torch.from_numpy(v).to(device) for k,v in arrays.items()}
