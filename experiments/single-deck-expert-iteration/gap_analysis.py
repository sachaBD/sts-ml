"""Read-only statistics on existing monitor trajectories; no search or training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import pyarrow.parquet as pq
import torch
from agents.combat.pv.model import PolicyValue,NAMES,WIDTHS

CARD_NAMES={11:'Anger',107:'Demon Form',83:'Corruption',124:'Dual Wield',33:'Bite',329:'Sword Boomerang',311:'Spot Weakness',179:'Headbutt',104:'Defend',25:'Bash',197:'Intimidate',65:'Clash',67:'Cleave'}

def token(row,action):
    i=row['moves'].index(action)
    return row['actions'][i*261:(i+1)*261]

def describe(row,action):
    t=token(row,action)
    name=CARD_NAMES.get(int(t[8]),str(int(t[8])))
    return name if t[0]==0 else 'end turn' if t[0]==4 else 'select '+name

def inputs(row):
    return [torch.tensor(row[n],dtype=torch.float32).reshape(1,-1,w) if n!='context' else torch.tensor(row[n],dtype=torch.float32).reshape(1,w) for n,w in zip(NAMES,WIDTHS)]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();p=a.out
    torch.set_num_threads(1)
    data={}
    for name,sub in [('teacher','reference-monitor'),('learned','iter015/eval')]:
        fights={r['fight_id']:r for r in pq.read_table(p/sub/'fights-0.parquet').to_pylist()}
        by={}
        for r in pq.read_table(p/'gap-investigation'/sub.replace('/','-')/'rows.parquet').to_pylist():by.setdefault(r['fight_id'],[]).append(r)
        for rows in by.values():rows.sort(key=lambda r:r['step'])
        data[name]=(fights,by)
    teacher,learner=data['teacher'][0],data['learned'][0]
    groups={k:('both_win' if teacher[k]['won'] and learner[k]['won'] else 'teacher_only' if teacher[k]['won'] else 'learner_only' if learner[k]['won'] else 'both_loss') for k in teacher}
    ckpt=torch.load(p/'iter015/model/model.pt',map_location='cpu',weights_only=False)
    net=PolicyValue(ckpt['width'],ckpt['value_activation']);net.load_state_dict(ckpt['state_dict']);net.eval()
    result=dict(paired_counts=dict(Counter(groups.values())),groups={},first_divergences=[],root_calibration={})
    for group in sorted(set(groups.values())):
        summary={}
        for name,(fights,by) in data.items():
            stats=[]
            for fid,g in groups.items():
                if g!=group:continue
                rows=by[fid];f=fights[fid];plays=[]
                for row in rows:
                    t=token(row,f['actions'][row['step']])
                    if t[0]==0:plays.append((int(t[8]),row['turn']))
                counts=Counter(c for c,t in plays)
                d=[t for c,t in plays if c==107];cor=[t for c,t in plays if c==83]
                stats.append(dict(fight_id=fid,demon_turn=d[0] if d else None,corruption_turn=cor[0] if cor else None,anger_plays=counts[11],boomerang_plays=counts[329],last_turn=rows[-1]['turn']))
            summary[name]=dict(n=len(stats),demon_played=sum(s['demon_turn'] is not None for s in stats),corruption_played=sum(s['corruption_turn'] is not None for s in stats),mean_anger_plays=statistics.mean(s['anger_plays'] for s in stats),mean_boomerang_plays=statistics.mean(s['boomerang_plays'] for s in stats),fights=stats)
        result['groups'][group]=summary
    search={(r['fight_id'],r['step']):r for r in pq.read_table(p/'iter015/eval/search-0.parquet').to_pylist()}
    with torch.no_grad():
        roots=[]
        for fid,rows in data['learned'][1].items():
            value,_=net(*inputs(rows[0]));roots.append(dict(fight_id=fid,group=groups[fid],raw_value=value.item(),search_value=rows[0]['root_value'],won=learner[fid]['won']))
        result['root_calibration']=dict(n=len(roots),mean_raw_value=statistics.mean(r['raw_value'] for r in roots),mean_search_value=statistics.mean(r['search_value'] for r in roots),observed_win_rate=sum(r['won'] for r in roots)/len(roots),by_group={g:dict(n=sum(r['group']==g for r in roots),mean_raw_value=statistics.mean(r['raw_value'] for r in roots if r['group']==g),mean_search_value=statistics.mean(r['search_value'] for r in roots if r['group']==g)) for g in sorted(set(groups.values()))})
        for fid,group in groups.items():
            if group!='teacher_only':continue
            step=next(i for i,(x,y) in enumerate(zip(teacher[fid]['actions'],learner[fid]['actions'])) if x!=y)
            row=data['learned'][1][fid][step];ta=teacher[fid]['actions'][step];la=learner[fid]['actions'][step]
            value,logits=net(*inputs(row));prior=logits.softmax(-1).flatten().tolist()
            children={c['action']:c for c in search.get((fid,step),{}).get('children',[])}
            result['first_divergences'].append(dict(fight_id=fid,step=step,turn=row['turn'],raw_value=value.item(),teacher_move=dict(bits=ta,name=describe(row,ta),prior=prior[row['moves'].index(ta)],search=children.get(ta)),learner_move=dict(bits=la,name=describe(row,la),prior=prior[row['moves'].index(la)],search=children.get(la))))
    dest=p/'gap-investigation/analysis.json';dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result['root_calibration'],indent=2))
    print('First divergences (not causal mistakes; duplicate-card ordering may be equivalent):')
    for r in result['first_divergences']:print(r['fight_id'].split(':')[-1],r['teacher_move']['name'],round(r['teacher_move']['prior'],3),'vs',r['learner_move']['name'],round(r['learner_move']['prior'],3))
    print(dest)

if __name__=='__main__':main()
