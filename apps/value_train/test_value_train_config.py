import dataclasses
import unittest
from pathlib import Path

from apps.value_train.train import make_train_config

CONFIG = {
    "run": {"id": "x"},
    "data": {"query": "select * from combat_v3", "oracle": False},
    "model": {"kind": "deep_sets_v1", "width": 64},
    "train": {"epochs": 1, "batch_size": 8, "lr": 0.001, "weight_decay": 0.0, "seed": 0, "label": "blend",
              "blend": 0.5, "lr_schedule": "constant", "keep": "last", "threads": 1, "device": "cpu",
              "aux_won_weight": 0.0, "aux_hp_weight": 0.0, "validation_fraction": 0.2},
}


def config(**train):
    return {**CONFIG, "train": {k: v for k, v in {**CONFIG["train"], **train}.items() if v is not None}}


class TrainConfigTests(unittest.TestCase):
    def test_complete_config(self):
        c = make_train_config(config(), Path("out"))
        self.assertEqual(c.output, Path("out/value_checkpoint.pt"))
        self.assertIsNone(c.initial_checkpoint)
        self.assertEqual(dataclasses.asdict(c)["blend"], 0.5)

    def test_missing_key_exits(self):
        with self.assertRaises(SystemExit):
            make_train_config(config(epochs=None), Path("out"))
        with self.assertRaises(SystemExit):
            make_train_config({**CONFIG, "data": {"query": "q"}}, Path("out"))

    def test_setting_that_does_not_apply(self):
        with self.assertRaisesRegex(ValueError, "blend"):
            make_train_config(config(label="terminal"), Path("out"))
        with self.assertRaisesRegex(ValueError, "blend"):
            make_train_config(config(blend=None), Path("out"))
        with self.assertRaisesRegex(ValueError, "split"):
            make_train_config(config(split="fresh"), Path("out"))

    def test_types(self):
        with self.assertRaisesRegex(TypeError, "epochs"):
            make_train_config(config(epochs=1.5), Path("out"))
        with self.assertRaisesRegex(TypeError, "lr"):
            make_train_config(config(lr=True), Path("out"))
        make_train_config(config(lr=1), Path("out"))  # an int is a float

    def test_model_needs_kind(self):
        with self.assertRaisesRegex(ValueError, "kind"):
            make_train_config({**config(), "model": {"width": 64}}, Path("out"))


if __name__ == "__main__":
    unittest.main()
