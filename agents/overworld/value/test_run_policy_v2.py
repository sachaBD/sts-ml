"""Current save/load round-trips and compatibility with pre-split path scorers."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from agents.overworld.value.core import encode, load_model, save_model
from agents.overworld.value.run_policy_v2 import RunPolicyV2


def batch():
    card = {"card_id": 15, "upgraded": 0, "misc": 0}
    state = {"deck": [card, {**card, "card_id": 25}], "relics": [], "potions": [],
             "hp": 30, "max_hp": 80, "gold": 90, "floor": 15, "potion_capacity": 3,
             "map": {"paths": [{"rooms": "M" * 14 + "R"}, {"rooms": "?" * 14 + "R"}]}}
    return encode([(state, [card, {**card, "card_id": 100}])], ["hexaghost"])


class RunPolicyV2CheckpointTests(unittest.TestCase):
    def test_round_trip(self):
        torch.manual_seed(42)
        for use_map, deck_attn, aux in [(True, 0, 0), (True, 1, 3), (False, 1, 0)]:
            with self.subTest(use_map=use_map, deck_attn=deck_attn, aux=aux), tempfile.TemporaryDirectory() as d:
                model = RunPolicyV2(width=8, hidden=16, depth=1, heads=2,
                                    use_map=use_map, deck_attn=deck_attn, aux=aux).eval()
                expected = model(batch())
                path = Path(d) / "model.pt"
                save_model(model, path)
                loaded = load_model(path)
                self.assertEqual(loaded.args, model.args)
                self.assertEqual(len(loaded(batch())), len(expected))
                for actual, target in zip(loaded(batch()), expected):
                    torch.testing.assert_close(actual, target, rtol=0, atol=0)

    def test_legacy_score_mlp(self):
        torch.manual_seed(7)
        model = RunPolicyV2(width=8, hidden=16, depth=1, heads=2).eval()
        legacy = model.state_dict().copy()
        legacy["score_mlp.0.weight"] = torch.cat([legacy.pop("score_path.weight"),
                                                   legacy.pop("score_ctx.weight")], dim=1)
        legacy["score_mlp.0.bias"] = legacy.pop("score_path.bias")
        legacy["score_mlp.2.weight"] = legacy.pop("score_out.weight")
        legacy["score_mlp.2.bias"] = legacy.pop("score_out.bias")
        args = {k: v for k, v in model.args.items() if k != "aux"}  # old checkpoints predate aux
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "legacy.pt"
            torch.save({"kind": model.KIND, "args": args, "state_dict": legacy}, path)
            loaded = load_model(path)
            torch.testing.assert_close(loaded(batch())[0], model(batch())[0], rtol=0, atol=0)
            paths, contexts = torch.randn(5, 8), torch.randn(5, 8)
            old_scorer = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
            old_scorer.load_state_dict({k.removeprefix("score_mlp."): v for k, v in legacy.items()
                                       if k.startswith("score_mlp.")})
            split = loaded.score_out(torch.relu(loaded.score_path(paths) + loaded.score_ctx(contexts)))
            torch.testing.assert_close(split, old_scorer(torch.cat([paths, contexts], dim=1)), rtol=1e-5, atol=1e-6)
            # Loading doesn't rename the caller's weights, and re-saving uses the current format.
            self.assertIn("score_mlp.0.weight", legacy)
            self.assertNotIn("score_path.weight", legacy)
            save_model(loaded, path)
            torch.testing.assert_close(load_model(path)(batch())[0], loaded(batch())[0], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
