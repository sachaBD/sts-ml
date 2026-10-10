"""Independent metric checks, ONNX parity, and descriptive strata. No optimizer or gameplay."""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import onnxruntime as ort
from agents.combat.pv.data import Shard
from agents.combat.pv.model import PolicyValue,NAMES
B=Path('runs/schema=combat_v4/date=2026-10-06/id=rollout-expert-iteration-v1/out')
O=Path('runs/schema=combat_v4/date=2026-10-07/id=expert-champ-absorption-audit-v1/out')
torch.set_num_threads(2)
a=json.loads((O/'results.json').read_text()); rows=pq.read_table(O/'per-state.parquet').to_pylist()
out={'onnx_parity':[],'strata':[],'split_checks':[]}
for u in (1,2,3):
    split=json.loads((B/f'strong/update{u}/split.json').read_text()); assert set(split.values())=={'train'}
    out['split_checks'].append({'update':u,'fights':len(split),'all_train':True})
    s=Shard(B/f'strong/update{u}/rows.parquet',False,split)
    for j in (u-1,u):
        path=Path(a['provenance']['model_paths'][j]);ck=torch.load(path,map_location='cpu',weights_only=False)
        net=PolicyValue(ck['width'],ck['value_activation']);net.load_state_dict(ck['state_dict']);net.eval()
        idx=np.linspace(0,len(s.rows)-1,64,dtype=int); b=s.batch(idx)
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        sess=ort.InferenceSession(str(path.with_suffix('.onnx')),sess_options=opts,providers=['CPUExecutionProvider'])
        got=sess.run(None,{n:b[0][n].numpy() for n in NAMES})
        with torch.inference_mode(): ref=net(*(b[0][n] for n in NAMES))
        diffs=[float(np.max(np.abs(x-y.numpy()))) for x,y in zip(got,ref)]
        assert diffs[0]<1e-3 and diffs[1]<1e-3,diffs
        out['onnx_parity'].append({'batch':u,'model':j,'states':64,'max_abs_value_logit':diffs})
    for label,pred in [('all',lambda r:True),('confidence_lt_.2',lambda r:r['confidence']<.2),('confidence_ge_.8',lambda r:r['confidence']>=.8),('turn_le_2',lambda r:r['turn']<=2),('loss',lambda r:not r['won'])]:
        for j in (u-1,u):
            rr=[r for r in rows if r['batch']==u and r['model']==j and r['has_policy'] and pred(r)]
            c=np.array([r['confidence'] for r in rr]); kl=np.array([r['kl'] for r in rr]);floor=np.array([r['floor'] for r in rr])
            out['strata'].append({'batch':u,'model':j,'stratum':label,'policy_rows':len(rr),'fights':len(set(r['fight_id'] for r in rr)),'row_mean_kl':float(kl.mean()),'row_mean_floor':float(floor.mean()),'confidence_weighted_kl':float(np.sum(c*kl)/c.sum()),'top1':float(np.mean([r['top1'] for r in rr]))})
(O/'verification.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
