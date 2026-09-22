import unittest

import torch
from sts_combat_rl.models.deep_sets import DeepSetsValue
from sts_combat_rl.training.data import collate_states, episode_split
from sts_combat_rl.training.evaluate_value import (
    boss_slice_masks,
    validate_checkpoint_source,
)


def row(ep, card=1):
    return {
        "encoding_version": 3,
        "episode_id": ep,
        "mcts_value": 0.2,
        "global_numeric": [0.0] * 50,
        "input_state": 1,
        "card_selection_task": 0,
        "cards": [
            {
                "card_id": card,
                "zone": 0,
                "card_type": 0,
                "target_type": 0,
                "numeric": [0.0] * 14,
            }
        ],
        "monsters": [{"monster_id": 1, "move_id": 1, "numeric": [0.0] * 9}],
        "card_monster_interactions": [
            {"card_index": 0, "monster_index": 0, "numeric": [0.0] * 6}
        ],
    }


class ValueTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        rs = [row(i) for i in range(10) for _ in range(2)]
        a = episode_split(rs, seed=4)
        b = episode_split(rs, seed=4)
        self.assertEqual(a[2:], b[2:])
        self.assertFalse(set(a[2]) & set(a[3]))

    def test_collate_offsets_and_batched_equivalence(self):
        rs = [row(0, 2), row(1, 3)]
        b = collate_states(rs)
        self.assertEqual(b["interaction_cards"].tolist(), [0, 1])
        self.assertEqual(b["interaction_monsters"].tolist(), [0, 1])
        torch.manual_seed(1)
        m = DeepSetsValue()
        batch = m(**{k: v for k, v in b.items() if k != "target"})
        singles = [
            m(**{k: v for k, v in collate_states([r]).items() if k != "target"})
            for r in rs
        ]
        self.assertTrue(torch.allclose(batch, torch.cat(singles)))

    def test_tiny_batch_overfits(self):
        torch.manual_seed(3)
        rows = [row(i, i + 1) for i in range(4)]
        for i, item in enumerate(rows):
            item["mcts_value"] = -0.75 + i * 0.5
            item["global_numeric"][0] = float(i)
        batch = collate_states(rows)
        model = DeepSetsValue(width=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        initial = (
            (
                (
                    model(**{k: v for k, v in batch.items() if k != "target"})
                    - batch["target"]
                )
                ** 2
            )
            .mean()
            .detach()
            .item()
        )
        for _ in range(150):
            optimizer.zero_grad()
            loss = (
                (
                    model(**{k: v for k, v in batch.items() if k != "target"})
                    - batch["target"]
                )
                ** 2
            ).mean()
            loss.backward()
            optimizer.step()
        final = (
            (
                (
                    model(**{k: v for k, v in batch.items() if k != "target"})
                    - batch["target"]
                )
                ** 2
            )
            .mean()
            .detach()
            .item()
        )
        self.assertLess(final, initial * 0.2)
        self.assertLess(final, 0.03)

    def test_boss_slice_masks_and_checksum_rejection(self):
        rows = []
        for hp in (0.8, 0.5, 0.2, None):
            item = row(len(rows))
            item["monsters"] = (
                []
                if hp is None
                else [{"monster_id": 9, "move_id": 1, "numeric": [0.0, hp] + [0.0] * 7}]
            )
            rows.append(item)
        masks = boss_slice_masks(rows, 9)
        self.assertEqual([int(mask.sum()) for mask in masks.values()], [1, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate_checkpoint_source(
                {
                    "source_sha256": "not-a-real-digest",
                    "encoding_version": 3,
                    "target_name": "mcts_value",
                },
                __file__,
            )

    def test_permutation_invariant(self):
        r = row(0)
        r["cards"].append(
            {
                "card_id": 4,
                "zone": 1,
                "card_type": 1,
                "target_type": 0,
                "numeric": [1.0] * 14,
            }
        )
        r["card_monster_interactions"].append(
            {"card_index": 1, "monster_index": 0, "numeric": [0.0] * 6}
        )
        q = dict(r)
        q["cards"] = list(reversed(r["cards"]))
        q["card_monster_interactions"] = [
            {
                "card_index": 1 - x["card_index"],
                "monster_index": x["monster_index"],
                "numeric": x["numeric"],
            }
            for x in r["card_monster_interactions"]
        ]
        m = DeepSetsValue()
        self.assertTrue(
            torch.allclose(
                m(**{k: v for k, v in collate_states([r]).items() if k != "target"}),
                m(**{k: v for k, v in collate_states([q]).items() if k != "target"}),
            )
        )


if __name__ == "__main__":
    unittest.main()
