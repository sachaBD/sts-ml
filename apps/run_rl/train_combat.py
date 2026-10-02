"""Fine-tune the frozen combat architecture on recent combat_v4 derived encodings."""
import argparse
import json
from pathlib import Path
from agents.combat.value.train_value import TrainConfig, run
from agents.combat.value.export_value_weights import export


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', nargs='+', required=True, help='combat_v4 run IDs')
    p.add_argument('--init', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--epochs', type=int, default=5)
    a = p.parse_args()
    arch = json.loads(a.init.with_suffix('.json').read_text())['architecture']
    ids = ','.join("'" + r.replace("'", "''") + "'" for r in a.data)
    sql = f"select * from combat_v4_training where run_id in ({ids}) and row_kind='decision'"
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(data=sql, oracle=False, model=arch, output=a.out/'value_checkpoint.pt',
        epochs=a.epochs, batch_size=1024, lr=1e-4, weight_decay=.01, seed=0, label='terminal', blend=None,
        lr_schedule='cosine', keep='best', threads=1, device='cuda', aux_won_weight=.01, aux_hp_weight=.1,
        initial_checkpoint=a.init, split='stable', validation_fraction=.1, corrections=None,
        correction_weight=None, row_weighting=None, aux_keep_weight=None, policy_weight=None)
    checkpoint = run(cfg)
    export(cfg.output, a.out/'value_weights.bin')
    (a.out/'summary.json').write_text(json.dumps({'checkpoint':'value_checkpoint.pt',
        'source_runs':a.data, 'metrics':checkpoint['metrics']}, indent=2))

if __name__ == '__main__':
    main()
