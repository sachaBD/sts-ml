"""Command-construction checks only: no stages, workers or training launched."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
SPEC=importlib.util.spec_from_file_location('champ_exit',Path(__file__).with_name('exit.py'))
exit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(exit)
class ExitFlagsTest(unittest.TestCase):
    def commands(self,flags,rounds=1):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for r in range(1,rounds+1):
                run=root/f'runs/schema=combat_v4/date=test/id=champ-ox-d-r{r:02d}'
                (run/'rows').mkdir(parents=True);(run/'rows/rows.parquet').touch()
            stages={}
            argv=['exit.py','--tag','d','--init',str(root/'init'),'--rounds',str(rounds),'--n','3000',
                  '--sims','64','--date','test','--worker','build/frozen/pv_worker.turn-targets-dc8f9579d2b6',*flags]
            def record(run,name,cmd):
                stages[name]=list(map(str,cmd))
                stages[(run.name,name)]=stages[name]
            with patch.object(exit,'ROOT',root),patch.object(exit.shutil,'copy2') as copy,patch.object(exit,'stage',side_effect=record),patch('sys.argv',argv):
                exit.main()
            source,frozen=copy.call_args.args
            self.assertEqual(str(source),'build/frozen/pv_worker.turn-targets-dc8f9579d2b6')
            for name in ['play','bench-oracle','bench-real','bench-policy','encode']:
                cmd=stages[name];self.assertEqual(cmd[cmd.index('--worker')+1],str(frozen))
            return stages
    def test_empty_unchanged(self):
        s=self.commands([])
        self.assertEqual(s['play'][-3:],['--oracle','--explore','--sample-turns'])
        self.assertEqual(s['bench-oracle'][-1],'--oracle')
        self.assertEqual(s['bench-policy'][-1],'--policy-only')
        self.assertEqual(s['bench-real'][s['bench-real'].index('--sims')+1],'2000')
        for name in ['bench-real','bench-oracle','bench-policy']:
            self.assertEqual(s[name][s[name].index('--starts')+1],str(exit.BENCH))
        self.assertEqual(s['compare'][2],str(exit.TEACHER))
        train=s['train'];self.assertEqual(len(train[train.index('--data')+1:train.index('--out')]),1)
    def test_split_and_stage_scoping(self):
        s=self.commands(['--selfplay-flags','--turn-search --turn-targets',
                         '--oracle-bench-flags','--turn-search',
                         '--real-flags','--turn-search --particles "4"'])
        self.assertEqual(s['play'][-5:],['--oracle','--explore','--sample-turns','--turn-search','--turn-targets'])
        self.assertEqual(s['bench-oracle'][-2:],['--oracle','--turn-search'])
        self.assertEqual(s['bench-real'][-3:],['--turn-search','--particles','4'])
        self.assertEqual(s['bench-real'][s['bench-real'].index('--sims')+1],'2000')
        self.assertNotIn('--turn-search',s['bench-policy'])
    def test_anchor_rows_every_round_and_real_only_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchors=[Path(tmp)/'c16.parquet',Path(tmp)/'c17.parquet']
            for path in anchors:path.touch()
            starts=Path(tmp)/'bench2k.parquet';baseline=Path(tmp)/'d5-baseline'
            s=self.commands(['--extra-rows',*map(str,anchors),'--real-starts',str(starts),
                             '--real-baseline',str(baseline)],rounds=4)
            for r in (1,2,3,4):
                cmd=s[(f'id=champ-ox-d-r{r:02d}','train')]
                data=cmd[cmd.index('--data')+1:cmd.index('--out')]
                self.assertEqual(len(data),min(r,3)+len(anchors))
                self.assertEqual(data[-2:],list(map(str,anchors)))
            self.assertEqual(s['bench-real'][s['bench-real'].index('--starts')+1],str(starts))
            self.assertEqual(s['bench-real'][s['bench-real'].index('--sims')+1],'2000')
            self.assertEqual(s['compare'][2],str(baseline))
            for name in ['bench-oracle','bench-policy']:
                self.assertEqual(s[name][s[name].index('--starts')+1],str(exit.BENCH))
    def test_missing_anchor_before_snapshot_or_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('sys.argv',['exit.py','--tag','e','--init','/none','--rounds','1','--n','1',
                                  '--sims','64','--extra-rows',str(Path(tmp)/'missing.parquet')]),\
                 patch.object(exit.shutil,'copy2') as copy,patch.object(exit,'stage') as stage:
                with self.assertRaises(SystemExit):exit.main()
                copy.assert_not_called();stage.assert_not_called()
    def test_bad_quoting_before_snapshot(self):
        with patch('sys.argv',['exit.py','--tag','d','--init','/none','--rounds','1','--n','1','--sims','64','--real-flags','"']),patch.object(exit.shutil,'copy2') as copy:
            with self.assertRaises(SystemExit):exit.main()
            copy.assert_not_called()
if __name__=='__main__':unittest.main()
