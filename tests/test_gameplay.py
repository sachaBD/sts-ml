import io
import unittest

from sts_combat_rl.training.evaluate_gameplay import aggregate
from sts_combat_rl.training.neural_gameplay import exact_read


class ShortReader(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        return super().read(min(size, 1))


class GameplayTests(unittest.TestCase):
    def test_exact_read_handles_short_reads(self):
        self.assertEqual(exact_read(ShortReader(b"abcd"), 4), b"abcd")

    def test_aggregate(self):
        rows = [
            {
                "won": True,
                "final_hp": 4,
                "split_hp": 3,
                "decisions": 2,
                "elapsed_seconds": 1.0,
            },
            {
                "won": False,
                "final_hp": 2,
                "split_hp": 1,
                "decisions": 4,
                "elapsed_seconds": 3.0,
            },
        ]
        value = aggregate(rows)
        self.assertEqual(value["win_rate"], 0.5)
        self.assertEqual(value["final_hp"], 3)
        self.assertEqual(value["split_count"], 2)


if __name__ == "__main__":
    unittest.main()
