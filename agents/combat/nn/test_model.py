"""Architecture specification and unchanged prediction contract."""
import io
import unittest

import torch

from agents.combat.nn.model import ACTION, WIDTH, Net


class ModelTests(unittest.TestCase):
    def test_architectures_and_checkpoint(self):
        torch.set_num_threads(1)
        for spec in (dict(dim=64, layers=2, heads=2), dict(dim=128, layers=4, heads=4)):
            with self.subTest(spec=spec):
                net = Net(**spec).eval()
                tokens = torch.zeros(2, 5, WIDTH)
                tokens[:, 0, 0] = 1
                tokens[:, 1:3, 0] = ACTION
                value, logits = net(tokens)
                self.assertEqual(value.shape, (2,))
                self.assertEqual(logits.shape, (2, 5))
                self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(((value >= 0) & (value <= 1)).all())
                self.assertTrue((logits[:, 3:] < -1e8).all())
                storage = io.BytesIO()
                torch.save(dict(spec=net.spec, net=net.state_dict()), storage)
                storage.seek(0)
                checkpoint = torch.load(storage, weights_only=True)
                restored = Net(**checkpoint['spec']).eval()
                restored.load_state_dict(checkpoint['net'])
                actual = restored(tokens)
                torch.testing.assert_close(value, actual[0])
                torch.testing.assert_close(logits, actual[1])

    def test_invalid_spec(self):
        for spec in (dict(dim=0), dict(layers=0), dict(heads=0), dict(dim=65, heads=2)):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                Net(**spec)


if __name__ == '__main__':
    unittest.main()
