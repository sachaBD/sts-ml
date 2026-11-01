"""Prediction invariants shared by both first-class combat architectures."""
import io
import unittest
from dataclasses import asdict

import torch

from .model import ACTION, WIDTH, ModelSpec, build_model


def tokens():
    x = torch.zeros(2, 16, WIDTH)
    x[:, :10, 0] = torch.tensor([1, 3, 5, 2, 2, 2, 2, ACTION, ACTION, ACTION])
    x[:, 4:7, 3] = torch.tensor([1, 2, 3])
    x[:, :10, 1] = torch.arange(10)
    x[:, :10, 4:] = torch.rand(2, 10, WIDTH - 4)
    return x


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_prediction_contract_padding_action_alignment_and_gradients(self):
        for kind in ('transformer', 'entity'):
            with self.subTest(kind=kind):
                net = build_model(ModelSpec(kind, 16, 2, 2)).eval()
                x = tokens()
                value, logits = net(x)
                self.assertEqual(value.shape, (2,))
                self.assertEqual(logits.shape, (2, 16))
                self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(((value >= 0) & (value <= 1)).all())
                self.assertTrue(torch.isfinite(logits[:, 7:10]).all())
                self.assertTrue((logits[x[..., 0] != ACTION] < -1e8).all())
                short = net(x[:, :10])
                torch.testing.assert_close(value, short[0])
                torch.testing.assert_close(logits[:, :10], short[1])
                permutation = [0, 1, 6, 3, 4, 5, 2, 9, 7, 8, *range(10, 16)]
                permuted = net(x[:, permutation])
                torch.testing.assert_close(value, permuted[0])
                torch.testing.assert_close(logits[:, permutation], permuted[1])
                with torch.autocast('cpu', torch.bfloat16):
                    mixed_value, mixed_logits = net(x)
                self.assertEqual(mixed_value.dtype, torch.float32)
                self.assertEqual(mixed_logits.dtype, torch.float32)
                self.assertTrue(torch.isfinite(mixed_logits).all())
                self.assertTrue((mixed_logits[x[..., 0] != ACTION] < -1e8).all())
                loss = -value.log().mean() - logits[:, 7:10].log_softmax(-1).mean()
                loss.backward()
                for parameter in net.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_spec_and_weights_reconstruct_predictions(self):
        for kind in ('transformer', 'entity'):
            with self.subTest(kind=kind):
                net = build_model(ModelSpec(kind, 16, 2, 2)).eval()
                x = tokens()
                storage = io.BytesIO()
                torch.save(dict(spec=asdict(net.spec), net=net.state_dict()), storage)
                storage.seek(0)
                checkpoint = torch.load(storage, weights_only=True)
                restored = build_model(ModelSpec(**checkpoint['spec'])).eval()
                restored.load_state_dict(checkpoint['net'])
                torch.testing.assert_close(net(x), restored(x))

    def test_invalid_architectures_are_rejected(self):
        for args in [('unknown', 16, 2, 2), ('entity', 0, 2, 2),
                     ('transformer', 16, 0, 2), ('entity', 16, 2, 0), ('transformer', 15, 2, 2)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                ModelSpec(*args)


if __name__ == '__main__':
    unittest.main()
