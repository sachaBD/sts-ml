"""Entity pooling, compaction and unchanged action-logit contract."""
import io
import unittest
import torch
from .entity import EntityNet
from .model import WIDTH

class EntityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.net = EntityNet(32, 2, 2).eval()
        self.x = torch.zeros(3, 24, WIDTH)
        self.x[:, :12, 0] = torch.tensor([1, 3, 5, 2, 2, 2, 2, 2, 2, 6, 6, 6])
        self.x[:, 5:9, 3] = torch.tensor([1, 1, 2, 3])
        self.x[:, :12, 1] = torch.arange(12)
        self.x[:, :12, 4:] = torch.rand(3, 12, WIDTH-4)

    def test_compaction_padding_and_order(self):
        with torch.no_grad():
            v, l = self.net(self.x)
            self.net.attended = 8
            vc, lc = self.net(self.x)
            torch.testing.assert_close(v, vc)
            torch.testing.assert_close(l, lc)
            self.net.attended = None
            vp, lp = self.net(self.x[:, :12])
            torch.testing.assert_close(v, vp)
            torch.testing.assert_close(l[:, :12], lp)
            perm = [0,1,8,3,4,7,6,5,2,9,10,11]+list(range(12,24))
            vo, lo = self.net(self.x[:,perm])
            torch.testing.assert_close(v, vo)
            torch.testing.assert_close(l[:,9:12], lo[:,9:12])
        self.assertTrue(torch.isfinite(l[:,9:12]).all())
        self.assertTrue((l[:,:9] < -1e8).all())

    def test_overflow_is_explicit(self):
        self.net.attended = 4
        self.assertTrue(torch.isnan(self.net(self.x)[0]).all())

    def test_gradients_and_checkpoint(self):
        self.net.train()
        v,l = self.net(self.x)
        loss = -v.log().mean() - l[:,9:12].log_softmax(-1).mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.net.parameters() if p.grad is not None))
        buf=io.BytesIO()
        torch.save(dict(spec=self.net.spec,net=self.net.state_dict()),buf);buf.seek(0)
        c=torch.load(buf,weights_only=True);spec=dict(c['spec']);self.assertEqual(spec.pop('kind'),'entity')
        restored=EntityNet(**spec);restored.load_state_dict(c['net'])
        torch.testing.assert_close(self.net(self.x), restored(self.x))

    def test_invalid_spec(self):
        for kw in [dict(heads=0),dict(dim=0),dict(layers=0),dict(dim=31,heads=2)]:
            with self.assertRaises(ValueError): EntityNet(**kw)

if __name__ == '__main__': unittest.main()
