"""Small adapter/split/paired-score checks; no combat generation."""
from pathlib import Path
from unittest.mock import patch

import unittest
import torch

from models.combat_outcome import learn_marginals as m


def natural(seed):
    return dict(run_seed=seed, category='boss', encounter='hexaghost', won=True, final_hp=20,
                pre=dict(hp=60, max_hp=80, potion_capacity=2,
                         deck=[dict(card_id=1, upgraded=0, misc=0)], relics=[], potions=[]))


def synthetic(group, donor=None, variant=0):
    r = natural(4)
    r.update(stage='boss' if donor is not None else 'easy', group_seed=group, kind='boss',
             variant=variant, seed=42, donor_run_seed=donor, battle_final_hp=20, start_hp=60)
    return r


def test_split_and_dedup():
    rs = [synthetic(100, 14), synthetic(101, 18), synthetic(102, 19), synthetic(109)]
    with patch.object(m, 'load', return_value=([natural(18), natural(14), natural(19)], {})), \
         patch.object(Path, 'glob', return_value=[Path('fake.parquet')]), \
         patch.object(Path, 'read_text', return_value='{"status":"done"}'), \
         patch.object(m, 'parquet_rows', return_value=iter(rs)):
        rows, _ = m.dataset('natural/x/x', ['marginal/x/x'])
    assert [r['split'] for r in rows] == ['dev', 'train', 'stop', 'train', 'dev', 'stop', 'stop']
    duplicate = synthetic(100, 14)
    with patch.object(m, 'load', return_value=([], {})), \
         patch.object(Path, 'glob', return_value=[Path('fake.parquet')]), \
         patch.object(Path, 'read_text', return_value='{"status":"done"}'), \
         patch.object(m, 'parquet_rows', return_value=iter([duplicate, duplicate.copy()])):
        with unittest.TestCase().assertRaisesRegex(ValueError, 'Duplicate'):
            m.dataset('natural/x/x', ['marginal/x/x'])


def test_paired_score_and_seed_guard():
    rows = [synthetic(10,14), synthetic(10,14,1)]
    for r in rows: r.update(domain='synthetic')
    rows[0].update(won=False, final_hp=0)
    rows[1]['final_hp'] = 23
    p = torch.tensor([0.,1.]); h = torch.zeros(2,m.BINS); h[:,4]=1
    centers = torch.arange(m.BINS)*5.+3
    result=m.paired(rows,p,h,centers)['hexaghost']
    assert result['score_delta_mae']==0
    assert result['zero_effect_mae']==53
    rows[1]['seed']=43
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Unmatched'):
        m.paired(rows,p,h,centers)


def test_gpu_or_cpu_fit():
    torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    rows=[natural(14),natural(15)]
    rows[1].update(won=False,final_hp=0)
    for r in rows:r.update(domain='natural')
    model=m.CombatOutcomeV1(encounters=2,width=16,hidden=32,dropout=.3)
    assert m.fit(model,rows,rows,{'hexaghost':1},device,1)==1
    p,h=m.infer(model,m.batch(rows,{'hexaghost':1},'cpu'),device)
    assert p.shape==(2,) and h.shape==(2,m.BINS)
    assert torch.isfinite(h).all()

def test_fixed_evaluation():
    with patch.object(m, 'load', return_value=([], {})), \
         patch.object(Path, 'glob', return_value=[Path('fake.parquet')]), \
         patch.object(Path, 'read_text', return_value='{"status":"done"}'), \
         patch.object(m, 'parquet_rows', side_effect=[
             iter([synthetic(100, 18), synthetic(101, 19), synthetic(102, 14)]),
             iter([synthetic(200, 18), synthetic(201, 19), synthetic(202, 14)])]):
        rows, _ = m.dataset('natural/x/x', ['marginal/x/first','marginal/x/later'], ['marginal/x/first'])
    assert [(r['group_seed'], r['split']) for r in rows] == [(100,'dev'),(101,'stop'),(102,'train'),(202,'train')]


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(f) for f in (test_split_and_dedup, test_paired_score_and_seed_guard, test_gpu_or_cpu_fit, test_fixed_evaluation))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
