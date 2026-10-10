import unittest
from apps.run_rl.teacher_correction import select_failures

class TeacherCorrection(unittest.TestCase):
    def test_only_losses_selected_without_duplicates_deterministically(self):
        ids=[str(i) for i in range(200)]
        results={fid:dict(status='completed',fight={'won':int(fid)>=150}) for fid in ids}
        selected=select_failures(ids,results)
        self.assertEqual(len(set(selected)),100)
        self.assertEqual(selected,select_failures(list(reversed(ids)),results))
        self.assertTrue(all(int(fid)<150 for fid in selected))

    def test_insufficient_losses_halts(self):
        with self.assertRaises(ValueError):select_failures(['a'],{'a':dict(status='completed',fight={'won':True})})

if __name__=='__main__':unittest.main()
