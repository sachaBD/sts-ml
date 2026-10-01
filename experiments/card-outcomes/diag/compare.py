"""Compact checkpoint comparison: Brier/HP MAE per domain/category + unbiased card-effect dMSE vs zero. Usage: compare.py MODEL_ID..."""
import json,sys,glob,subprocess
for mid in sys.argv[1:]:
    d=glob.glob(f'runs/schema=combat_outcome_v1/date=*/id={mid}/out')[0]
    r=json.load(open(d+'/report.json'))['results']
    print(f'### {mid}  best_epoch nat={r["natural"].get("best_epoch")} aug={r["augmented"].get("best_epoch")}')
    for k in sorted(r['natural']['metrics']):
        n=r['natural']['metrics'][k]; a=r['augmented']['metrics'][k]
        print(f'  {k:16s} fights={n["fights"]:5d} Brier {n["brier"]:.4f}->{a["brier"]:.4f}  HP MAE {n["hp_mae"]:.2f}->{a["hp_mae"]:.2f}')
    for arm in ('natural','augmented'):
        out=subprocess.run([sys.executable,'experiments/card-outcomes/diag/reliability.py',f'{d}/{arm}-predictions.jsonl'],capture_output=True,text=True).stdout.split('\n')
        i=[j for j,l in enumerate(out) if l.startswith('---')][0]
        print(f'  card dMSE [{arm}]: '+' | '.join(l for l in out[i+1:] if l))
