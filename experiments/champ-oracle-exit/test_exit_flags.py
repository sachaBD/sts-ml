"""Command-construction checks only: no stages, workers or training launched."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
SPEC=importlib.util.spec_from_file_location('champ_exit',Path(__file__).with_name('exit.py'))
exit=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(exit)
class ExitFlagsTest(unittest.TestCase):
    def commands(self,flags):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'runs/schema=combat_v4/date=test/id=champ-ox-d-r01'
            (run/'rows').mkdir(parents=True);(run/'rows/rows.parquet').touch()
            stages={}
            argv=['exit.py','--tag','d','--init',str(root/'init'),'--rounds','1','--n','3000',
                  '--sims','64','--date','test','--worker','build/frozen/pv_worker.turn-targets-dc8f9579d2b6',*flags]
            with patch.object(exit,'ROOT',root),patch.object(exit.shutil,'copy2') as copy,patch.object(exit,'stage',side_effect=lambda run,name,cmd:stages.update({name:list(map(str,cmd))})),patch('sys.argv',argv):
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
    def test_split_and_stage_scoping(self):
        s=self.commands(['--selfplay-flags','--turn-search --turn-targets',
                         '--oracle-bench-flags','--turn-search',
                         '--real-flags','--turn-search --particles "4"'])
        self.assertEqual(s['play'][-5:],['--oracle','--explore','--sample-turns','--turn-search','--turn-targets'])
        self.assertEqual(s['bench-oracle'][-2:],['--oracle','--turn-search'])
        self.assertEqual(s['bench-real'][-3:],['--turn-search','--particles','4'])
        self.assertEqual(s['bench-real'][s['bench-real'].index('--sims')+1],'2000')
        self.assertNotIn('--turn-search',s['bench-policy'])
    def test_bad_quoting_before_snapshot(self):
        with patch('sys.argv',['exit.py','--tag','d','--init','/none','--rounds','1','--n','1','--sims','64','--real-flags','"']),patch.object(exit.shutil,'copy2') as copy:
            with self.assertRaises(SystemExit):exit.main()
            copy.assert_not_called()
if __name__=='__main__':unittest.main()
