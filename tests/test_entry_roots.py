import unittest

from sts_combat_rl.data.entry_roots import deck_signature_split


class DeckSignatureSplitTests(unittest.TestCase):
    def test_replicates_and_rows_do_not_leak(self):
        rows = [
            {"deck_signature": signature, "episode_id": episode, "replicate": replicate}
            for signature, episode in (("a", 0), ("a", 1), ("b", 2), ("b", 3))
            for replicate in (0, 1)
        ]
        train, valid, train_ids, valid_ids = deck_signature_split(rows, 0.5, 7)
        self.assertFalse(set(train_ids) & set(valid_ids))
        self.assertFalse(
            {r["deck_signature"] for r in train} & {r["deck_signature"] for r in valid}
        )
        self.assertEqual({r["deck_signature"] for r in train + valid}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
